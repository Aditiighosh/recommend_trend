const express = require("express");
const axios = require("axios");
const router = express.Router();

// POST /clothing/recommendations - Fetch recommendations from Python service
router.post("/recommendations", async (req, res) => {
    const { detectedClothing, aesthetic, occasion } = req.body;

    try {
        const response = await axios.post("http://localhost:5000/recommendations", {
            detected_clothing: detectedClothing,
            aesthetic,
            occasion,
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.error("Error fetching recommendations:", error.message);
        res.status(500).json({ error: "Failed to fetch recommendations" });
    }
});

module.exports = router;