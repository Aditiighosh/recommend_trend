const express = require("express");
const bodyParser = require("body-parser");
const clothingRoutes = require("./routes/clothingRoutes");
const accessoriesRoutes = require("./routes/accessoriesRoutes");
const recommendationRoutes = require("./routes/recommendationRoutes");
const multer = require("multer");
const fs = require("fs");
const axios = require("axios");

const app = express();

// Middleware
app.use(bodyParser.json());

// Home Route (NEWLY ADDED)
app.get("/", (req, res) => {
  res.send("Welcome to the Virtual Dressing Room API!");
});

// Configure Multer for image uploads
const upload = multer({ dest: "backend/uploads/" });

// Routes
app.use("/clothing", clothingRoutes);
app.use("/accessories", accessoriesRoutes);
app.use("/recommendations", recommendationRoutes);

// Image Upload and Processing Route
app.post("/upload", upload.single("image"), async (req, res) => {
  const imagePath = req.file.path;

  try {
    // Send image to Flask service
    const formData = new FormData();
    formData.append("image", fs.createReadStream(imagePath));

    const response = await axios.post("http://localhost:5000/process-image", formData, {
      headers: formData.getHeaders(),
    });

    // Return Flask service's response to the client
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error processing image:", error.message);
    res.status(500).json({ error: "Failed to process image" });
  } finally {
    // Clean up uploaded file
    fs.unlinkSync(imagePath);
  }
});

// Recommendations Route
app.post("/recommendations", async (req, res) => {
  try {
    // Send the detected clothing items, aesthetic, and occasion to the Flask service
    const { detectedClothing, aesthetic, occasion } = req.body;

    const response = await axios.post("http://localhost:5000/recommendations", {
      detected_clothing: detectedClothing,
      aesthetic: aesthetic,
      occasion: occasion,
    });

    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching recommendations:", error.message);
    res.status(500).json({ error: "Failed to fetch recommendations" });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
