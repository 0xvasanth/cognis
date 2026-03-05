//! Prebuilt agent graphs.
//!
//! This module provides ready-to-use agent graph constructors,
//! starting with the classic ReAct (Reasoning + Acting) agent pattern.

pub mod react_agent;

pub use react_agent::create_react_agent;
