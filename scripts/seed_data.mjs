import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const dataDir = path.join(projectRoot, "flask", "data");

const salesPath = path.join(projectRoot, "seeFood_Dashboard", "lib", "sales-data.js");
const salesModule = await import(pathToFileURL(salesPath));

fs.mkdirSync(dataDir, { recursive: true });

fs.writeFileSync(
  path.join(dataDir, "sales-data.json"),
  JSON.stringify(salesModule.salesData, null, 2)
);

fs.writeFileSync(
  path.join(dataDir, "prediction-data.json"),
  JSON.stringify(salesModule.predictionData, null, 2)
);

console.log("Seed sales fallback data written to", dataDir);
