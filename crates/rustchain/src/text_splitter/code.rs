/// Programming languages with known code separators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Python,
    JavaScript,
    TypeScript,
    Rust,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Scala,
    Swift,
    Markdown,
    Html,
    LaTeX,
    Sol,
}

impl Language {
    /// Get the ordered list of separators for this language.
    pub fn get_separators(&self) -> Vec<String> {
        match self {
            Language::Python => vec![
                "\nclass ", "\ndef ", "\n\tdef ",
                "\n\n", "\n", " ", "",
            ],
            Language::JavaScript | Language::TypeScript => vec![
                "\nfunction ", "\nconst ", "\nlet ", "\nvar ",
                "\nclass ", "\nif ", "\nfor ", "\nwhile ",
                "\n\n", "\n", " ", "",
            ],
            Language::Rust => vec![
                "\nfn ", "\nstruct ", "\nenum ", "\nimpl ",
                "\ntrait ", "\npub fn ", "\npub struct ",
                "\npub enum ", "\npub trait ",
                "\nmod ", "\npub mod ",
                "\n\n", "\n", " ", "",
            ],
            Language::Go => vec![
                "\nfunc ", "\ntype ", "\nvar ", "\nconst ",
                "\n\n", "\n", " ", "",
            ],
            Language::Java => vec![
                "\nclass ", "\npublic ", "\nprotected ", "\nprivate ",
                "\nstatic ", "\nif ", "\nfor ", "\nwhile ",
                "\n\n", "\n", " ", "",
            ],
            Language::C | Language::Cpp => vec![
                "\nstruct ", "\nvoid ", "\nint ", "\nfloat ",
                "\nclass ", "\n#include ", "\n#define ",
                "\n\n", "\n", " ", "",
            ],
            Language::Ruby => vec![
                "\ndef ", "\nclass ", "\nmodule ",
                "\n\n", "\n", " ", "",
            ],
            Language::Scala => vec![
                "\nclass ", "\nobject ", "\ndef ",
                "\nval ", "\nvar ",
                "\n\n", "\n", " ", "",
            ],
            Language::Swift => vec![
                "\nfunc ", "\nclass ", "\nstruct ", "\nenum ",
                "\nprotocol ", "\nextension ",
                "\n\n", "\n", " ", "",
            ],
            Language::Markdown => vec![
                "\n## ", "\n### ", "\n#### ", "\n##### ",
                "\n\n", "\n", " ", "",
            ],
            Language::Html => vec![
                "<div", "<p", "<br", "<li",
                "<h1", "<h2", "<h3", "<h4",
                "<table", "<tr", "<td",
                "\n\n", "\n", " ", "",
            ],
            Language::LaTeX => vec![
                "\n\\chapter{", "\n\\section{", "\n\\subsection{",
                "\n\\subsubsection{", "\n\\begin{",
                "\n\n", "\n", " ", "",
            ],
            Language::Sol => vec![
                "\ncontract ", "\nfunction ", "\nmodifier ",
                "\nevent ", "\nstruct ", "\nenum ",
                "\n\n", "\n", " ", "",
            ],
        }
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }
}
