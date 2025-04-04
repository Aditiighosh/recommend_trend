import { PrismaClient } from "@prisma/client";
import { withOptimize } from "@prisma/extension-optimize";

const prisma = new PrismaClient().$extends(
  withOptimize({ apiKey: process.env.OPTIMIZE_API_KEY || "default_api_key" })
)

export default prisma;
