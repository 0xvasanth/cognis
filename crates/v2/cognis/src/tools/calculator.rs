//! Math expression evaluator.
//!
//! Supports `+ - * / % ^`, parentheses, unary `-`, and floating-point
//! numbers. Pure Rust, no deps. Returns the result as text.
//!
//! Intentionally no variables / no functions / no code execution — keep
//! it predictable for an LLM-driven agent. Need symbolic math? Pick a
//! different tool.

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

/// Math expression input.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CalculatorInput {
    /// The expression to evaluate. Example: `"(2 + 3) * 4"`.
    pub expression: String,
}

/// Stateless calculator tool.
#[derive(Debug, Default, Clone, Copy)]
pub struct Calculator;

impl Calculator {
    /// Construct a `Calculator`.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate a numeric expression. Supports +, -, *, /, %, ^, parentheses, \
         and unary minus. Returns the result as a number."
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(CalculatorInput)).unwrap_or_default())
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let v = input.into_json();
        let parsed: CalculatorInput = serde_json::from_value(v).map_err(|e| {
            CognisError::ToolValidationError(format!("calculator: invalid args: {e}"))
        })?;
        let result = evaluate(&parsed.expression).map_err(|e| CognisError::Tool {
            name: "calculator".into(),
            reason: e,
        })?;
        Ok(ToolOutput::Text(format_number(result)))
    }
}

/// Format `f64` losslessly when integer, otherwise with up to 12 sig figs.
fn format_number(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n:.12}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

// ---------------------------------------------------------------------------
// Recursive-descent expression parser.
//
// Grammar (in precedence order, weakest first):
//   expr   := term ( ('+' | '-') term )*
//   term   := factor ( ('*' | '/' | '%') factor )*
//   factor := unary ( '^' factor )?         // right-associative
//   unary  := '-' unary | atom
//   atom   := number | '(' expr ')'
// ---------------------------------------------------------------------------

fn evaluate(input: &str) -> std::result::Result<f64, String> {
    let mut p = Parser::new(input);
    let v = p.parse_expr()?;
    p.skip_ws();
    if p.pos < p.bytes.len() {
        return Err(format!(
            "unexpected character `{}` at position {}",
            p.bytes[p.pos] as char, p.pos
        ));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn consume(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    left += self.parse_term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    left -= self.parse_term()?;
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_term(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    left *= self.parse_factor()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let r = self.parse_factor()?;
                    if r == 0.0 {
                        return Err("division by zero".into());
                    }
                    left /= r;
                }
                Some(b'%') => {
                    self.pos += 1;
                    let r = self.parse_factor()?;
                    if r == 0.0 {
                        return Err("modulo by zero".into());
                    }
                    left %= r;
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_factor(&mut self) -> std::result::Result<f64, String> {
        let base = self.parse_unary()?;
        if self.consume(b'^') {
            let exp = self.parse_factor()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> std::result::Result<f64, String> {
        if self.consume(b'-') {
            let v = self.parse_unary()?;
            return Ok(-v);
        }
        if self.consume(b'+') {
            return self.parse_unary();
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> std::result::Result<f64, String> {
        if self.consume(b'(') {
            let v = self.parse_expr()?;
            if !self.consume(b')') {
                return Err(format!("expected `)` at position {}", self.pos));
            }
            return Ok(v);
        }
        self.parse_number()
    }

    fn parse_number(&mut self) -> std::result::Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        let mut saw_digit = false;
        let mut saw_dot = false;
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b'0'..=b'9' => {
                    saw_digit = true;
                    self.pos += 1;
                }
                b'.' if !saw_dot => {
                    saw_dot = true;
                    self.pos += 1;
                }
                b'e' | b'E' => {
                    self.pos += 1;
                    if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
        if !saw_digit {
            return Err(format!("expected number at position {start}"));
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| format!("non-utf8 number: {e}"))?;
        slice
            .parse::<f64>()
            .map_err(|e| format!("invalid number `{slice}`: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn arithmetic_precedence() {
        assert_eq!(evaluate("1 + 2 * 3").unwrap(), 7.0);
        assert_eq!(evaluate("(1 + 2) * 3").unwrap(), 9.0);
        assert_eq!(evaluate("2 ^ 3 ^ 2").unwrap(), 512.0); // right-assoc
        assert_eq!(evaluate("10 % 3").unwrap(), 1.0);
        assert_eq!(evaluate("-3 + 5").unwrap(), 2.0);
        assert_eq!(evaluate("--5").unwrap(), 5.0);
        assert_eq!(evaluate("2.5 * 4").unwrap(), 10.0);
        assert_eq!(evaluate("1e3 / 100").unwrap(), 10.0);
    }

    #[test]
    fn errors_on_bad_input() {
        assert!(evaluate("").is_err());
        assert!(evaluate("1 / 0").is_err());
        assert!(evaluate("(1 + 2").is_err());
        assert!(evaluate("1 + abc").is_err());
        assert!(evaluate("1 2").is_err());
    }

    #[tokio::test]
    async fn tool_runs_via_trait() {
        let t = Calculator::new();
        let mut args = std::collections::HashMap::new();
        args.insert("expression".into(), json!("(2 + 3) * 4"));
        let out = t._run(ToolInput::Structured(args)).await.unwrap();
        assert_eq!(out.as_string(), "20");
    }

    #[tokio::test]
    async fn tool_validation_error_on_missing_field() {
        let t = Calculator::new();
        let args = std::collections::HashMap::new();
        let err = t._run(ToolInput::Structured(args)).await.unwrap_err();
        assert_eq!(err.category(), "tool_validation");
    }

    #[test]
    fn format_number_handles_integers_and_floats() {
        assert_eq!(format_number(20.0), "20");
        assert_eq!(format_number(2.5), "2.5");
        assert_eq!(format_number(1.0 / 3.0), "0.333333333333");
    }
}
