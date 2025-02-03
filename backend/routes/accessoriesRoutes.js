const express = require("express");
const router = express.Router();
const pool = require("../db");

// Add an accessory
router.post("/", async (req, res) => {
  const { type, tags, compatible_with, aesthetic, trending, occasion } = req.body;
  try {
    const result = await pool.query(
      "INSERT INTO accessories (type, tags, compatible_with, aesthetic, trending, occasion) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
      [type, tags, compatible_with, aesthetic, trending, occasion]
    );
    res.status(201).json(result.rows[0]);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "Error adding accessory" });
  }
});

// Get all accessories
router.get("/", async (req, res) => {
  try {
    const result = await pool.query("SELECT * FROM accessories");
    res.status(200).json(result.rows);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "Error retrieving accessories" });
  }
});

module.exports = router;
