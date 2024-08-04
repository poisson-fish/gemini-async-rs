
use std::collections::HashMap;
use reqwest::Client;
use tokio::sync::Mutex;
mod types;
use types::*;

const BASE_URL: &str = "https://{location}-aiplatform.googleapis.com/v1beta1/projects";

pub struct GeminiClient {
    client: Client,
    project_id: String,
    model: String,
    location: String,
    api_key: String,
    functions: Mutex<HashMap<String, Box<dyn Fn(HashMap<String, String>) -> Result<String, String> + Send + Sync>>>,
}

impl GeminiClient {
    pub fn builder() -> GeminiClientBuilder {
        GeminiClientBuilder::default()
    }

    pub async fn count_tokens(&self, request: CountTokensRequest) -> Result<CountTokensResponse, reqwest::Error> {
        let url = format!("{}/{}/locations/{}/models/{}:countTokens?key={}", BASE_URL, self.project_id, self.location, self.model, self.api_key);
        let response = self.client.post(&url).json(&request).send().await?;
        response.json().await
    }

    pub async fn generate_content(&self, request: GenerateContentRequest) -> Result<GenerateContentResponse, reqwest::Error> {
        let url = format!("{}/{}/locations/{}/models/{}:generateContent?key={}", BASE_URL, self.project_id, self.location, self.model, self.api_key);
        let response = self.client.post(&url).json(&request).send().await?;
        response.json().await
    }

    // Helper functions to create content parts
    pub fn create_text_content(role: String, text: String) -> Content {
        Content {
            role,
            parts: vec![Part::Text(text)],
        }
    }

    pub fn create_function_call_content(role: String, name: String, args: HashMap<String, String>) -> Content {
        Content {
            role,
            parts: vec![Part::FunctionCall { name, args }],
        }
    }

    // Helper function to create function declarations
    pub fn create_function_declaration(name: String, description: String, parameters: FunctionParameters) -> FunctionDeclaration {
        FunctionDeclaration {
            name,
            description,
            parameters,
        }
    }

    // Register a function
    pub async fn register_function<F>(&self, name: String, function: F)
    where
        F: Fn(HashMap<String, String>) -> Result<String, String> + Send + Sync + 'static,
    {
        let mut functions = self.functions.lock().await;
        functions.insert(name, Box::new(function));
    }

    // Function to execute function calls
    #[allow(dead_code)]
    async fn execute_function_call(&self, function_call: &Part) -> Result<String, String> {
        match function_call {
            Part::FunctionCall { name, args } => {
                // Find the registered function
                let functions = self.functions.lock().await;
                if let Some(function) = functions.get(name) {
                    function(args.clone())
                } else {
                    Err(format!("Unknown function: {}", name))
                }
            }
            _ => Err("Not a function call".to_string()),
        }
    }
}

#[derive(Default)]
pub struct GeminiClientBuilder {
    client: Option<Client>,
    project_id: Option<String>,
    model: Option<String>,
    location: Option<String>,
    api_key: Option<String>,
}

impl GeminiClientBuilder {
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn project_id(mut self, project_id: String) -> Self {
        self.project_id = Some(project_id);
        self
    }

    pub fn model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    pub fn api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    pub fn build(self) -> Result<GeminiClient, String> {
        match (self.client, self.project_id, self.model, self.location, self.api_key) {
            (Some(client), Some(project_id), Some(model), Some(location), Some(api_key)) => Ok(GeminiClient {
                client,
                project_id,
                model,
                location,
                api_key,
                functions: Mutex::new(HashMap::new()),
            }),
            _ => Err("Missing required fields to build GeminiClient".to_string()),
        }
    }
}