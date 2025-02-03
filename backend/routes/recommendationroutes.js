const express = require("express");
const router = express.Router();
const pool = require("../db");

// Get recommendations
router.post("/", async (req, res) => {
  const { type, color, aesthetic, occasion } = req.body;

  try {
    const result = await pool.query(
      "SELECT * FROM accessories WHERE $1 = ANY(compatible_with) OR aesthetic = $2 OR occasion = $3",
      [type, aesthetic, occasion]
    );
    res.status(200).json(result.rows);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "Error retrieving recommendations" });
  }
});

module.exports = router;
