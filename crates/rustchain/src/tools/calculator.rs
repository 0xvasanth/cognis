//! A safe math expression evaluator tool.
//!
//! Supports +, -, *, /, parentheses, and correct operator precedence via a
//! recursive descent parser. No `eval` or code execution is used.

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde_json::{json, Value};

/// A safe math expression evaluator.
pub struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate mathematical expressions. Input should be a valid math expression like '2 + 3 * 4'."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let expression = extract_expression(&input)?;
        let result = evaluate(&expression).map_err(|e| {
            RustChainError::ToolException(format!("Failed to evaluate expression: {e}"))
        })?;
        Ok(ToolOutput::Content(Value::String(format_result(result))))
    }
}

/// Extract the expression string from various input formats.
fn extract_expression(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(expr)) = map.get("expression") {
                Ok(expr.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'expression'".into(),
                ))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(expr)) = tc.args.get("expression") {
                Ok(expr.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'expression'".into(),
                ))
            }
        }
    }
}

/// Format a floating-point result, removing trailing zeros for clean output.
fn format_result(value: f64) -> String {
    if value == value.floor() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{}", value)
    }
}

// ---------------------------------------------------------------------------
// Recursive descent expression parser
// ---------------------------------------------------------------------------

/// Token types produced by the tokenizer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

/// Tokenize an expression string into a sequence of tokens.
fn tokenize(input: &str) -> std::result::Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                // Determine if this is a unary minus (negative number) or subtraction.
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::LParen)
                    );

                if is_unary {
                    // Parse as a negative number
                    i += 1;
                    // Skip whitespace between minus and number/paren
                    while i < chars.len() && chars[i] == ' ' {
                        i += 1;
                    }
                    if i < chars.len() && chars[i] == '(' {
                        // -( ... ) => push -1 * (
                        tokens.push(Token::Number(-1.0));
                        tokens.push(Token::Star);
                        tokens.push(Token::LParen);
                        i += 1;
                    } else if i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        let start = i;
                        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                            i += 1;
                        }
                        let num_str: String = chars[start..i].iter().collect();
                        let num: f64 = num_str
                            .parse()
                            .map_err(|_| format!("Invalid number: -{num_str}"))?;
                        tokens.push(Token::Number(-num));
                    } else {
                        return Err("Unexpected character after unary minus".into());
                    }
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Invalid number: {num_str}"))?;
                tokens.push(Token::Number(num));
            }
            c => {
                return Err(format!("Unexpected character: '{c}'"));
            }
        }
    }

    Ok(tokens)
}

/// Parser state.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Parse the full expression.
    fn parse_expr(&mut self) -> std::result::Result<f64, String> {
        let result = self.parse_addition()?;
        if self.pos < self.tokens.len() {
            return Err(format!("Unexpected token: {:?}", self.tokens[self.pos]));
        }
        Ok(result)
    }

    /// Addition and subtraction (lowest precedence).
    fn parse_addition(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_multiplication()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Plus => {
                    self.next();
                    left += self.parse_multiplication()?;
                }
                Token::Minus => {
                    self.next();
                    left -= self.parse_multiplication()?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Multiplication and division.
    fn parse_multiplication(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_primary()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Star => {
                    self.next();
                    left *= self.parse_primary()?;
                }
                Token::Slash => {
                    self.next();
                    let right = self.parse_primary()?;
                    if right == 0.0 {
                        return Err("Division by zero".into());
                    }
                    left /= right;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Primary: numbers and parenthesized expressions.
    fn parse_primary(&mut self) -> std::result::Result<f64, String> {
        match self.next() {
            Some(Token::Number(n)) => Ok(n),
            Some(Token::LParen) => {
                let val = self.parse_addition()?;
                match self.next() {
                    Some(Token::RParen) => Ok(val),
                    _ => Err("Expected closing parenthesis".into()),
                }
            }
            Some(tok) => Err(format!("Unexpected token: {tok:?}")),
            None => Err("Unexpected end of expression".into()),
        }
    }
}

/// Evaluate a math expression string and return the result.
pub fn evaluate(expression: &str) -> std::result::Result<f64, String> {
    let tokens = tokenize(expression)?;
    if tokens.is_empty() {
        return Err("Empty expression".into());
    }
    let mut parser = Parser::new(tokens);
    parser.parse_expr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_addition() {
        let result = evaluate("2 + 3").unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_precedence() {
        let result = evaluate("2 + 3 * 4").unwrap();
        assert!((result - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_parentheses() {
        let result = evaluate("(2 + 3) * 4").unwrap();
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_division() {
        let result = evaluate("10 / 3").unwrap();
        assert!((result - 3.333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_negative() {
        let result = evaluate("-5 + 3").unwrap();
        assert!((result - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_complex() {
        let result = evaluate("((1 + 2) * (3 + 4)) / 7").unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_calculator_via_run_json() {
        let tool = CalculatorTool;
        let input = serde_json::json!({"expression": "2 + 3 * 4"});
        let result = tool.run_json(&input).await.unwrap();
        assert_eq!(result, Value::String("14".to_string()));
    }
}
