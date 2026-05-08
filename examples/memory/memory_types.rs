//! Tour of V2 memory types: Buffer (unbounded), Window (bounded FIFO),
//! TokenBufferMemory (token-budget). All implement the same Memory
//! trait so they slot into AgentBuilder::with_memory.

use cognis::prelude::*;
use cognis::{Buffer, Memory, TokenBufferMemory, Window};

fn main() {
    let mut buffer = Buffer::new();
    let mut window = Window::new(3);
    let mut tokens = TokenBufferMemory::new(20);

    for s in ["a", "bb", "ccc", "dddd", "eeeee"] {
        buffer.write(Message::human(s.to_string()));
        window.write(Message::human(s.to_string()));
        tokens.write(Message::human(s.to_string()));
    }
    println!(
        "Buffer:                {} messages (unbounded)",
        buffer.seed().len()
    );
    println!(
        "Window(3):             {} messages (last 3)",
        window.seed().len()
    );
    println!(
        "TokenBufferMemory(20): {} messages (≤20 chars)",
        tokens.seed().len()
    );
}
