use cognis_core::error::{ErrorCode, CognisError};

#[test]
fn error_code_as_str() {
    assert_eq!(
        ErrorCode::InvalidPromptInput.as_str(),
        "INVALID_PROMPT_INPUT"
    );
    assert_eq!(
        ErrorCode::InvalidToolResults.as_str(),
        "INVALID_TOOL_RESULTS"
    );
    assert_eq!(
        ErrorCode::MessageCoercionFailure.as_str(),
        "MESSAGE_COERCION_FAILURE"
    );
    assert_eq!(
        ErrorCode::ModelAuthentication.as_str(),
        "MODEL_AUTHENTICATION"
    );
    assert_eq!(ErrorCode::ModelNotFound.as_str(), "MODEL_NOT_FOUND");
    assert_eq!(ErrorCode::ModelRateLimit.as_str(), "MODEL_RATE_LIMIT");
    assert_eq!(
        ErrorCode::OutputParsingFailure.as_str(),
        "OUTPUT_PARSING_FAILURE"
    );
}

#[test]
fn error_code_equality() {
    assert_eq!(ErrorCode::ModelNotFound, ErrorCode::ModelNotFound);
    assert_ne!(ErrorCode::ModelNotFound, ErrorCode::ModelRateLimit);
}

#[test]
fn output_parser_error_display() {
    let err = CognisError::OutputParserError {
        message: "bad output".into(),
        observation: Some("hint".into()),
        llm_output: Some("raw".into()),
    };
    assert_eq!(err.to_string(), "Output parser error: bad output");
}

#[test]
fn context_overflow_display() {
    let err = CognisError::ContextOverflow("too many tokens".into());
    assert_eq!(err.to_string(), "Context overflow: too many tokens");
}

#[test]
fn other_error_display() {
    let err = CognisError::Other("something went wrong".into());
    assert_eq!(err.to_string(), "something went wrong");
}

#[test]
fn serde_json_error_converts() {
    let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
    let err: CognisError = json_err.into();
    assert!(matches!(err, CognisError::SerializationError(_)));
}

#[test]
fn result_type_alias_works() {
    let ok: cognis_core::error::Result<i32> = Ok(42);
    assert_eq!(ok.unwrap(), 42);

    let err: cognis_core::error::Result<i32> = Err(CognisError::Other("fail".into()));
    assert!(err.is_err());
}

#[test]
fn tool_exception_display() {
    let err = CognisError::ToolException("Calculator failed".into());
    assert_eq!(err.to_string(), "Tool exception: Calculator failed");
}

#[test]
fn tool_exception_matches() {
    let err = CognisError::ToolException("test".into());
    assert!(matches!(err, CognisError::ToolException(_)));
}
