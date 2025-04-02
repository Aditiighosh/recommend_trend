"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var client_1 = require("@prisma/client");
var extension_optimize_1 = require("@prisma/extension-optimize");
var prisma = new client_1.PrismaClient().$extends((0, extension_optimize_1.withOptimize)({ apiKey: process.env.OPTIMIZE_API_KEY || "default_api_key" }));
exports.default = prisma;
