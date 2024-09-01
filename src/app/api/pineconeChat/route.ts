import { PineconeStore } from "@langchain/pinecone";
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import {
    Message as VercelChatMessage,
    StreamingTextResponse,
    createStreamDataTransformer,
  } from "ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';
import { RunnableSequence } from "@langchain/core/runnables";

const TEMPLATE = `Answer the user's questions based only on the following context.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

const formatMessage = (message: VercelChatMessage) => {
    return `${message.role}: ${message.content}`;
  };

const pc = new Pinecone({
apiKey: process.env.PINECONE_API_KEY!
});

export async function POST(req: Request, res: Response) {
    try {
        const { messages } = await req.json();

        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

        const currentMessageContent = messages[messages.length - 1].content;

        const embeddings = new OpenAIEmbeddings({
            model: 'text-embedding-3-small',
            apiKey: process.env.OPEN_AI_API_KEY
            });
        
        const index = pc.Index('pinecone-chatbot');

        const llm = new ChatOpenAI({
            apiKey: process.env.OPEN_AI_API_KEY,
            model: 'gpt-4o-mini',
        });

        const parser = new HttpResponseOutputParser();
        
        const prompt = PromptTemplate.fromTemplate(TEMPLATE);
        
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: index
        });
        
        const retreiver = vectorStore.asRetriever();

        const chain = RunnableSequence.from([
            {
                question: (input) => input.question,
                chat_history: (input) => input.chat_history,
                context: async () => {
                    const results = await retreiver._getRelevantDocuments(currentMessageContent);
                    let formattedContext = '';
                    results.forEach((doc)=>{
                        formattedContext = `${formattedContext}, ${doc.pageContent}`
                    });
                    console.log(formattedContext);
                    return formattedContext
                }
            },
            prompt,
            llm,
            parser,
        ]);

        const stream = await chain.stream({
            chat_history: formattedPreviousMessages,
            question: currentMessageContent
        });


        return new StreamingTextResponse(
            stream.pipeThrough(createStreamDataTransformer())
        );

    } catch (error) {
        console.log(error);
    }
}