import 'dotenv/config';
import {PDFLoader} from '@langchain/community/document_loaders/fs/pdf';
import {RecursiveCharacterTextSplitter} from '@langchain/textsplitters';
import {GoogleGenerativeAIEmbeddings} from '@langchain/google-genai';
import { MongoClient } from 'mongodb';
import {MongoDBAtlasVectorSearch} from '@langchain/mongodb';

const loader = new PDFLoader('./iPhone_16e_PER_Feb2025.pdf')

const docs = await loader.load();
// console.log(docs.length);
// console.log(docs);

const textsplitters = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:200
})

const allSplits = await textsplitters.splitDocuments(docs);

// console.log(allSplits)

const embeddings = new GoogleGenerativeAIEmbeddings({
    model:'gemini-embedding-001'
});

// const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
// console.log(vector1);

const client      = new MongoClient(process.env.MONGO_URL || '');
const collection  = client.db('test').collection('testcollection');
const vectorStore = new MongoDBAtlasVectorSearch(embeddings,{
    collection:collection,
    indexName :'vector_test_index',
    textKey :'text',
    embeddingKey:'embedding'
})

const x = await vectorStore.addDocuments(allSplits);

const res = await vectorStore.similaritySearch('what is the iphone model?');
console.log(res);
await client.close();