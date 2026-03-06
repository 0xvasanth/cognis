use thiserror::Error;

/// Error codes matching langchain ErrorCode enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    InvalidPromptInput,
    InvalidToolResults,
    MessageCoercionFailure,
    ModelAuthentication,
    ModelNotFound,
    ModelRateLimit,
    OutputParsingFailure,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidPromptInput => "INVALID_PROMPT_INPUT",
            Self::InvalidToolResults => "INVALID_TOOL_RESULTS",
            Self::MessageCoercionFailure => "MESSAGE_COERCION_FAILURE",
            Self::ModelAuthentication => "MODEL_AUTHENTICATION",
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::ModelRateLimit => "MODEL_RATE_LIMIT",
            Self::OutputParsingFailure => "OUTPUT_PARSING_FAILURE",
        }
    }
}

/// Core error type for RustChain.
#[derive(Debug, Error)]
pub enum RustChainError {
    #[error("Output parser error: {message}")]
    OutputParserError {
        message: String,
        observation: Option<String>,
        llm_output: Option<String>,
    },

    #[error("Context overflow: {0}")]
    ContextOverflow(String),

    #[error("Tracer error: {0}")]
    TracerError(String),

    #[error("Invalid key: {0}")]
    InvalidKey(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Tool exception: {0}")]
    ToolException(String),

    #[error("Tool validation error: {0}")]
    ToolValidationError(String),

    #[error("Schema annotation error: {0}")]
    SchemaAnnotationError(String),

    #[error("Recursion limit exceeded: {0}")]
    RecursionLimitExceeded(String),

    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("HTTP error {status}: {body}")]
    HttpError { status: u16, body: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, RustChainError>;
