// 1. Import necessary modules and libraries
import { OpenAI } from "langchain/llms";
import { OpenAIChat } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as dotenv from "dotenv";

//seting up server
import express from "express";
const app = express();
app.set("view engine", "ejs");

app.use(express.urlencoded({ extended: true }));

app.listen(3000);

app.get("/", (req, res) => {
  res.render("./home");
});

app.post("/out", (req, res) => {
  // 2. Load environment variables
  dotenv.config();

  // 3. Set up input data and paths
  const txtFilename = "test";
  const question = req.body.question;
  const txtPath = `./${txtFilename}.txt`;
  const VECTOR_STORE_PATH = `${txtFilename}.index`;

  // 4. Define the main function runWithEmbeddings
  const runWithEmbeddings = async () => {
    // 5. Initialize the OpenAI model with an empty configuration object
    const model = new OpenAIChat({
      model: "gpt-4",
      max_tokens: 4096,
      temperature: 1,
    });

    // 6. Check if the vector store file exists
    let vectorStore;
    if (fs.existsSync(VECTOR_STORE_PATH)) {
      // 6.1. If the vector store file exists, load it into memory
      console.log("Vector Exists..");
      vectorStore = await HNSWLib.load(
        VECTOR_STORE_PATH,
        new OpenAIEmbeddings()
      );
    } else {
      // 6.2. If the vector store file doesn't exist, create it
      // 6.2.1. Read the input text file
      const text = fs.readFileSync(txtPath, "utf8");
      // 6.2.2. Create a RecursiveCharacterTextSplitter with a specified chunk size
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1500,
      });
      // 6.2.3. Split the input text into documents
      const docs = await textSplitter.createDocuments([text]);
      // 6.2.4. Create a new vector store from the documents using OpenAIEmbeddings
      vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
      // 6.2.5. Save the vector store to a file
      await vectorStore.save(VECTOR_STORE_PATH);
    }

    // 7. Create a RetrievalQAChain by passing the initialized OpenAI model and the vector store retriever
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    // 8. Call the RetrievalQAChain with the input question, and store the result in the 'res' variable
    const response = await chain.call({
      query: question,
    });
    if (
      response.text === "I don't know the answer." ||
      !response.text.trim() ||
      response.text === "I don't know."
    ) {
      response.text =
        "please call the Technology Solutions Center x3619 or visit us at BAC C107 for the help."; // Replace with your default answer
    }

    res.render("out", { question, ans: response.text });
  };

  // 10. Execute the main function runWithEmbeddings
  runWithEmbeddings();
});
