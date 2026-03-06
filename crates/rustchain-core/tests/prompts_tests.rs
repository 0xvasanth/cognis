use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::prompts::*;
use rustchain_core::runnables::Runnable;

// ─── PromptTemplate Tests ───

#[test]
fn test_prompt_template_from_template() {
    let pt = PromptTemplate::from_template("Hello {name}, you are {age} years old.");
    assert_eq!(pt.input_variables, vec!["name", "age"]);
}

#[test]
fn test_prompt_template_format() {
    let pt = PromptTemplate::from_template("Hello {name}!");
    let mut kwargs = HashMap::new();
    kwargs.insert("name".into(), json!("Alice"));
    let result = pt.format(&kwargs).unwrap();
    assert_eq!(result, "Hello Alice!");
}

#[test]
fn test_prompt_template_format_with_numbers() {
    let pt = PromptTemplate::from_template("{x} + {y} = {z}");
    let mut kwargs = HashMap::new();
    kwargs.insert("x".into(), json!(2));
    kwargs.insert("y".into(), json!(3));
    kwargs.insert("z".into(), json!(5));
    let result = pt.format(&kwargs).unwrap();
    assert_eq!(result, "2 + 3 = 5");
}

#[test]
fn test_prompt_template_escaped_braces() {
    let pt = PromptTemplate::from_template("Use {{braces}} with {var}.");
    assert_eq!(pt.input_variables, vec!["var"]);
    let mut kwargs = HashMap::new();
    kwargs.insert("var".into(), json!("value"));
    let result = pt.format(&kwargs).unwrap();
    assert_eq!(result, "Use {braces} with value.");
}

#[test]
fn test_prompt_template_missing_variable_error() {
    let pt = PromptTemplate::from_template("Hello {name}!");
    let kwargs = HashMap::new();
    assert!(pt.format(&kwargs).is_err());
}

#[test]
fn test_prompt_template_partial() {
    let pt = PromptTemplate::from_template("Hello {name}, age {age}.");
    let pt = pt.partial({
        let mut m = HashMap::new();
        m.insert("name".into(), PartialValue::Static(json!("Bob")));
        m
    });

    assert_eq!(pt.input_variables, vec!["age"]);

    let mut kwargs = HashMap::new();
    kwargs.insert("age".into(), json!(25));
    let result = pt.format(&kwargs).unwrap();
    assert_eq!(result, "Hello Bob, age 25.");
}

#[test]
fn test_prompt_template_dynamic_partial() {
    let pt = PromptTemplate::from_template("Time: {time}, Q: {question}");
    let pt = pt.partial({
        let mut m = HashMap::new();
        m.insert(
            "time".into(),
            PartialValue::Dynamic(Box::new(|| json!("12:00"))),
        );
        m
    });

    let mut kwargs = HashMap::new();
    kwargs.insert("question".into(), json!("What?"));
    let result = pt.format(&kwargs).unwrap();
    assert_eq!(result, "Time: 12:00, Q: What?");
}

#[test]
fn test_prompt_template_format_prompt() {
    let pt = PromptTemplate::from_template("Hello {name}!");
    let mut kwargs = HashMap::new();
    kwargs.insert("name".into(), json!("Alice"));
    let pv = pt.format_prompt(&kwargs).unwrap();
    assert_eq!(pv.to_string(), "Hello Alice!");
    assert_eq!(pv.to_messages().len(), 1);
}

#[tokio::test]
async fn test_prompt_template_runnable_invoke() {
    let pt = PromptTemplate::from_template("Say {what}");
    let result = pt.invoke(json!({"what": "hello"}), None).await.unwrap();
    assert_eq!(result, json!("Say hello"));
}

#[tokio::test]
async fn test_prompt_template_runnable_single_var() {
    let pt = PromptTemplate::from_template("Echo: {input}");
    let result = pt.invoke(json!("test"), None).await.unwrap();
    assert_eq!(result, json!("Echo: test"));
}

// ─── MessagePromptTemplate Tests ───

#[test]
fn test_message_prompt_template_human() {
    let mpt = MessagePromptTemplate::from_role("human", "Hello {name}!").unwrap();
    let mut kwargs = HashMap::new();
    kwargs.insert("name".into(), json!("Alice"));
    let msgs = mpt.format_messages(&kwargs).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content().text(), "Hello Alice!");
}

#[test]
fn test_message_prompt_template_system() {
    let mpt = MessagePromptTemplate::from_role("system", "You are {role}.").unwrap();
    let mut kwargs = HashMap::new();
    kwargs.insert("role".into(), json!("helpful"));
    let msgs = mpt.format_messages(&kwargs).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content().text(), "You are helpful.");
}

#[test]
fn test_message_prompt_template_ai() {
    let mpt = MessagePromptTemplate::from_role("ai", "I think {thought}").unwrap();
    let mut kwargs = HashMap::new();
    kwargs.insert("thought".into(), json!("therefore I am"));
    let msgs = mpt.format_messages(&kwargs).unwrap();
    assert_eq!(msgs[0].content().text(), "I think therefore I am");
}

#[test]
fn test_message_prompt_template_input_variables() {
    let mpt = MessagePromptTemplate::from_role("human", "{a} and {b}").unwrap();
    assert_eq!(mpt.input_variables(), &["a", "b"]);
}

// ─── MessagesPlaceholder Tests ───

#[test]
fn test_messages_placeholder_required() {
    let ph = MessagesPlaceholder::new("history");
    assert_eq!(ph.input_variables(), vec!["history"]);

    let kwargs = HashMap::new();
    let result = ph.format_messages(&kwargs);
    assert!(result.is_err()); // Missing required variable
}

#[test]
fn test_messages_placeholder_optional() {
    let ph = MessagesPlaceholder::new("history").optional(true);
    assert!(ph.input_variables().is_empty()); // Optional = no required vars

    let kwargs = HashMap::new();
    let msgs = ph.format_messages(&kwargs).unwrap();
    assert!(msgs.is_empty()); // Returns empty list
}

#[test]
fn test_messages_placeholder_with_messages() {
    let ph = MessagesPlaceholder::new("history");
    let mut kwargs = HashMap::new();
    kwargs.insert(
        "history".into(),
        json!([
            {"type": "human", "content": "Hi"},
            {"type": "ai", "content": "Hello!"}
        ]),
    );
    let msgs = ph.format_messages(&kwargs).unwrap();
    assert_eq!(msgs.len(), 2);
}

#[test]
fn test_messages_placeholder_n_messages() {
    let ph = MessagesPlaceholder::new("history").n_messages(1);
    let mut kwargs = HashMap::new();
    kwargs.insert(
        "history".into(),
        json!([
            {"type": "human", "content": "First"},
            {"type": "human", "content": "Second"},
            {"type": "human", "content": "Third"}
        ]),
    );
    let msgs = ph.format_messages(&kwargs).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content().text(), "Third"); // Last message
}

// ─── ChatPromptTemplate Tests ───

#[test]
fn test_chat_prompt_template_basic() {
    let cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are {persona}."),
        ("human", "{question}"),
    ])
    .unwrap();

    assert!(cpt.input_variables.contains(&"persona".to_string()));
    assert!(cpt.input_variables.contains(&"question".to_string()));
}

#[test]
fn test_chat_prompt_template_format_messages() {
    let cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are helpful."),
        ("human", "What is {topic}?"),
    ])
    .unwrap();

    let mut kwargs = HashMap::new();
    kwargs.insert("topic".into(), json!("Rust"));
    let msgs = cpt.format_messages(&kwargs).unwrap();

    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].content().text(), "You are helpful.");
    assert_eq!(msgs[1].content().text(), "What is Rust?");
}

#[test]
fn test_chat_prompt_template_with_placeholder() {
    let cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are helpful."),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])
    .unwrap();

    let mut kwargs = HashMap::new();
    kwargs.insert(
        "history".into(),
        json!([
            {"type": "human", "content": "Previous question"},
            {"type": "ai", "content": "Previous answer"}
        ]),
    );
    kwargs.insert("question".into(), json!("New question"));
    let msgs = cpt.format_messages(&kwargs).unwrap();

    assert_eq!(msgs.len(), 4); // system + 2 history + human
    assert_eq!(msgs[0].content().text(), "You are helpful.");
    assert_eq!(msgs[1].content().text(), "Previous question");
    assert_eq!(msgs[2].content().text(), "Previous answer");
    assert_eq!(msgs[3].content().text(), "New question");
}

#[test]
fn test_chat_prompt_template_partial() {
    let cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are {persona}."),
        ("human", "{question}"),
    ])
    .unwrap();

    let cpt = cpt.partial({
        let mut m = HashMap::new();
        m.insert("persona".into(), PartialValue::Static(json!("a teacher")));
        m
    });

    assert_eq!(cpt.input_variables, vec!["question"]);

    let mut kwargs = HashMap::new();
    kwargs.insert("question".into(), json!("What is 2+2?"));
    let msgs = cpt.format_messages(&kwargs).unwrap();
    assert_eq!(msgs[0].content().text(), "You are a teacher.");
}

#[tokio::test]
async fn test_chat_prompt_template_runnable_invoke() {
    let cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "Be concise."),
        ("human", "{query}"),
    ])
    .unwrap();

    let result = cpt
        .invoke(json!({"query": "Hello"}), None)
        .await
        .unwrap();

    let msgs: Vec<serde_json::Value> = serde_json::from_value(result).unwrap();
    assert_eq!(msgs.len(), 2);
}

#[test]
fn test_chat_prompt_template_append() {
    let mut cpt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are helpful."),
    ])
    .unwrap();

    cpt.append("human", "Ask about {topic}").unwrap();
    assert!(cpt.input_variables.contains(&"topic".to_string()));

    let mut kwargs = HashMap::new();
    kwargs.insert("topic".into(), json!("weather"));
    let msgs = cpt.format_messages(&kwargs).unwrap();
    assert_eq!(msgs.len(), 2);
}

// ─── FewShotPromptTemplate Tests ───

#[test]
fn test_few_shot_prompt_template() {
    let example_prompt = PromptTemplate::from_template("Q: {question}\nA: {answer}");
    let examples = vec![
        {
            let mut m = HashMap::new();
            m.insert("question".into(), json!("What is 2+2?"));
            m.insert("answer".into(), json!("4"));
            m
        },
        {
            let mut m = HashMap::new();
            m.insert("question".into(), json!("What is 3+3?"));
            m.insert("answer".into(), json!("6"));
            m
        },
    ];

    let fs = FewShotPromptTemplate::new(examples, example_prompt, "Q: {input}\nA:");

    let mut kwargs = HashMap::new();
    kwargs.insert("input".into(), json!("What is 4+4?"));
    let result = fs.format(&kwargs).unwrap();

    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("4"));
    assert!(result.contains("What is 3+3?"));
    assert!(result.contains("6"));
    assert!(result.contains("What is 4+4?"));
}

#[test]
fn test_few_shot_prompt_template_with_prefix() {
    let example_prompt = PromptTemplate::from_template("{input} -> {output}");
    let examples = vec![{
        let mut m = HashMap::new();
        m.insert("input".into(), json!("hello"));
        m.insert("output".into(), json!("HELLO"));
        m
    }];

    let fs = FewShotPromptTemplate::new(examples, example_prompt, "{text} ->")
        .with_prefix("Uppercase the input:");

    let mut kwargs = HashMap::new();
    kwargs.insert("text".into(), json!("world"));
    let result = fs.format(&kwargs).unwrap();

    assert!(result.starts_with("Uppercase the input:"));
    assert!(result.contains("hello -> HELLO"));
    assert!(result.ends_with("world ->"));
}

#[tokio::test]
async fn test_few_shot_prompt_template_runnable() {
    let example_prompt = PromptTemplate::from_template("{q}: {a}");
    let examples = vec![{
        let mut m = HashMap::new();
        m.insert("q".into(), json!("Hi"));
        m.insert("a".into(), json!("Hello"));
        m
    }];

    let fs = FewShotPromptTemplate::new(examples, example_prompt, "{input}:");
    let result = fs.invoke(json!({"input": "Bye"}), None).await.unwrap();
    let text = result.as_str().unwrap();
    assert!(text.contains("Hi: Hello"));
    assert!(text.contains("Bye:"));
}

// ─── BaseExampleSelector Tests ───

struct FixedExampleSelector {
    examples: Vec<HashMap<String, Value>>,
}

#[async_trait]
impl BaseExampleSelector for FixedExampleSelector {
    async fn select_examples(
        &self,
        _input: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        Ok(self.examples.clone())
    }

    async fn add_example(&self, _example: HashMap<String, Value>) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_few_shot_with_example_selector() {
    let selector = Arc::new(FixedExampleSelector {
        examples: vec![{
            let mut m = HashMap::new();
            m.insert("q".into(), json!("What?"));
            m.insert("a".into(), json!("That."));
            m
        }],
    });

    let example_prompt = PromptTemplate::from_template("Q: {q}\nA: {a}");
    let fs = FewShotPromptTemplate::with_example_selector(selector, example_prompt, "{input}");

    let mut kwargs = HashMap::new();
    kwargs.insert("input".into(), json!("test"));
    let result = fs.format_async(&kwargs).await.unwrap();
    assert!(result.contains("Q: What?"));
    assert!(result.contains("A: That."));
    assert!(result.contains("test"));
}

// ─── FewShotChatMessagePromptTemplate Tests ───

#[test]
fn test_few_shot_chat_message_template() {
    let example_prompt = MessagePromptTemplate::from_role("human", "{input}").unwrap();
    let examples = vec![
        {
            let mut m = HashMap::new();
            m.insert("input".into(), json!("example 1"));
            m
        },
        {
            let mut m = HashMap::new();
            m.insert("input".into(), json!("example 2"));
            m
        },
    ];

    let fsct = FewShotChatMessagePromptTemplate::new(examples, example_prompt);
    let kwargs = HashMap::new();
    let messages = fsct.format_messages(&kwargs).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content().text(), "example 1");
    assert_eq!(messages[1].content().text(), "example 2");
}

#[tokio::test]
async fn test_few_shot_chat_message_template_runnable() {
    let example_prompt = MessagePromptTemplate::from_role("human", "{text}").unwrap();
    let examples = vec![{
        let mut m = HashMap::new();
        m.insert("text".into(), json!("hello"));
        m
    }];

    let fsct = FewShotChatMessagePromptTemplate::new(examples, example_prompt);
    let result = fsct.invoke(json!({}), None).await.unwrap();
    let msgs: Vec<Value> = serde_json::from_value(result).unwrap();
    assert_eq!(msgs.len(), 1);
}

#[tokio::test]
async fn test_few_shot_chat_with_selector() {
    let selector = Arc::new(FixedExampleSelector {
        examples: vec![{
            let mut m = HashMap::new();
            m.insert("text".into(), json!("selected example"));
            m
        }],
    });

    let example_prompt = MessagePromptTemplate::from_role("human", "{text}").unwrap();
    let fsct =
        FewShotChatMessagePromptTemplate::with_example_selector(selector, example_prompt);

    let kwargs = HashMap::new();
    let messages = fsct.format_messages_async(&kwargs).await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content().text(), "selected example");
}

// ─── DictPromptTemplate Tests ───

#[test]
fn test_dict_prompt_template_basic() {
    let template = json!({
        "name": "{person_name}",
        "greeting": "Hello {person_name}!"
    });
    let dt = DictPromptTemplate::new(template);
    assert!(dt.input_variables.contains(&"person_name".to_string()));

    let mut kwargs = HashMap::new();
    kwargs.insert("person_name".into(), json!("Alice"));
    let result = dt.format(&kwargs).unwrap();
    assert_eq!(result["name"], "Alice");
    assert_eq!(result["greeting"], "Hello Alice!");
}

#[test]
fn test_dict_prompt_template_nested() {
    let template = json!({
        "user": {
            "name": "{name}",
            "age": 30
        },
        "tags": ["{tag1}", "{tag2}"]
    });
    let dt = DictPromptTemplate::new(template);
    assert_eq!(dt.input_variables.len(), 3);

    let mut kwargs = HashMap::new();
    kwargs.insert("name".into(), json!("Bob"));
    kwargs.insert("tag1".into(), json!("admin"));
    kwargs.insert("tag2".into(), json!("active"));
    let result = dt.format(&kwargs).unwrap();
    assert_eq!(result["user"]["name"], "Bob");
    assert_eq!(result["user"]["age"], 30);
    assert_eq!(result["tags"][0], "admin");
    assert_eq!(result["tags"][1], "active");
}

#[tokio::test]
async fn test_dict_prompt_template_runnable() {
    let template = json!({"key": "{value}"});
    let dt = DictPromptTemplate::new(template);
    let result = dt.invoke(json!({"value": "hello"}), None).await.unwrap();
    assert_eq!(result["key"], "hello");
}

// ─── ImagePromptTemplate Tests ───

#[test]
fn test_image_prompt_template_basic() {
    let it = ImagePromptTemplate::new("https://example.com/{image_id}.png");
    assert_eq!(it.input_variables, vec!["image_id"]);
    assert_eq!(it.detail, "auto");

    let mut kwargs = HashMap::new();
    kwargs.insert("image_id".into(), json!("cat"));
    let result = it.format(&kwargs).unwrap();
    assert_eq!(result["url"], "https://example.com/cat.png");
    assert_eq!(result["detail"], "auto");
}

#[test]
fn test_image_prompt_template_with_detail() {
    let it = ImagePromptTemplate::new("https://img.com/{id}")
        .with_detail("high");
    assert_eq!(it.detail, "high");

    let mut kwargs = HashMap::new();
    kwargs.insert("id".into(), json!("photo1"));
    let result = it.format(&kwargs).unwrap();
    assert_eq!(result["detail"], "high");
}

#[test]
fn test_image_prompt_template_no_variables() {
    let it = ImagePromptTemplate::new("https://example.com/static.png");
    assert!(it.input_variables.is_empty());

    let kwargs = HashMap::new();
    let result = it.format(&kwargs).unwrap();
    assert_eq!(result["url"], "https://example.com/static.png");
}

#[tokio::test]
async fn test_image_prompt_template_runnable() {
    let it = ImagePromptTemplate::new("https://cdn.com/{file}").with_detail("low");
    let result = it
        .invoke(json!({"file": "avatar.jpg"}), None)
        .await
        .unwrap();
    assert_eq!(result["url"], "https://cdn.com/avatar.jpg");
    assert_eq!(result["detail"], "low");
}
