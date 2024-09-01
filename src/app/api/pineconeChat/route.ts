// Importing necessary modules and classes from various libraries
import { PineconeStore } from "@langchain/pinecone"; // PineconeStore for managing vector-based search
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'; // OpenAI components for language model and embeddings
import {
    Message as VercelChatMessage, // Vercel's message type
    StreamingTextResponse, // Response type for streaming text
    createStreamDataTransformer, // Transformer for processing streaming data
} from "ai"; // AI library
import { Pinecone } from "@pinecone-database/pinecone"; // Pinecone client for vector database interactions
import { PromptTemplate } from '@langchain/core/prompts'; // Template class for prompt creation
import { HttpResponseOutputParser } from 'langchain/output_parsers'; // Parser for handling HTTP responses
import { RunnableSequence } from "@langchain/core/runnables"; // RunnableSequence for creating complex sequences of actions

// Define a template string for the prompt used in the language model
const TEMPLATE = `Answer the user's questions based only on the following context.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

// Function to format messages for logging or processing
const formatMessage = (message: VercelChatMessage) => {
    return `${message.role}: ${message.content}`;
};

// Initialize the Pinecone client with the API key from environment variables
const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY! // Non-null assertion, assumes API key is present
});

// Define an asynchronous function to handle POST requests
export async function POST(req: Request, res: Response) {
    try {
        // Parse incoming JSON request body to extract messages
        const { messages } = await req.json();

        // Format previous messages for use in the prompt
        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

        // Extract the content of the current message
        const currentMessageContent = messages[messages.length - 1].content;

        // Initialize OpenAI embeddings with model and API key
        const embeddings = new OpenAIEmbeddings({
            model: 'text-embedding-3-small', // Model to use for embeddings
            apiKey: process.env.OPEN_AI_API_KEY // API key for OpenAI
        });
        
        // Access the Pinecone index for 'pinecone-chatbot'
        const index = pc.Index('pinecone-chatbot');

        // Initialize the ChatOpenAI model with the specified model and API key
        const llm = new ChatOpenAI({
            apiKey: process.env.OPEN_AI_API_KEY,
            model: 'gpt-4o-mini', // Model to use for generating responses
        });

        // Create an instance of the HTTP response parser
        const parser = new HttpResponseOutputParser();
        
        // Create a prompt template instance using the defined template string
        const prompt = PromptTemplate.fromTemplate(TEMPLATE);
        
        // Initialize PineconeStore with existing index and embeddings
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: index
        });
        
        // Create a retriever from the vector store
        const retreiver = vectorStore.asRetriever();

        // Create a sequence of actions to process the request
        const chain = RunnableSequence.from([
            {
                // Define functions to extract input values and context
                question: (input) => input.question,
                chat_history: (input) => input.chat_history,
                context: async () => {
                    // Retrieve relevant documents based on the current message content
                    const results = await retreiver._getRelevantDocuments(currentMessageContent);
                    let formattedContext = '';
                    // Format the retrieved documents
                    results.forEach((doc)=>{
                        formattedContext = `${formattedContext}, ${doc.pageContent}`
                    });
                    console.log(formattedContext); // Log the formatted context
                    return formattedContext;
                }
            },
            prompt, // Use the prompt template
            llm, // Use the language model
            parser, // Use the response parser
        ]);

        // Stream the response from the chain
        const stream = await chain.stream({
            chat_history: formattedPreviousMessages,
            question: currentMessageContent
        });

        // Return a streaming response using the transformed data
        return new StreamingTextResponse(
            stream.pipeThrough(createStreamDataTransformer())
        );

    } catch (error) {
        // Log any errors that occur
        console.log(error);
    }
}
