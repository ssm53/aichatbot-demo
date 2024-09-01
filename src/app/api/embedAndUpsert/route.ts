import { OpenAIEmbeddings } from '@langchain/openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PineconeStore } from "@langchain/pinecone";
import klData from "../../../../data/klData.json"
import type { Document } from "@langchain/core/documents";
import { NextResponse } from 'next/server';


const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!
});

export async function POST() {
    try {
    // Create an index

    //     await pc.createIndex({
    //         name: 'pinecone-chatbot',
    //         dimension: 1536, // Replace with your model dimensions
    //         metric: 'cosine', // Replace with your model metric
    //         spec: { 
    //             serverless: { 
    //                 cloud: 'aws', 
    //                 region: 'us-east-1' 
    //             }
    //         } 
    //   });
        const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-3-small',
        apiKey: process.env.OPEN_AI_API_KEY
        });

        const index = pc.Index('pinecone-chatbot');

        await PineconeStore.fromTexts(klData, {}, embeddings, {
            pineconeIndex: index
        })

        console.log("Success")
        return NextResponse.json("success")

    } catch (error) {
        console.log(error);
        return NextResponse.json(error)

    }
}