const express = require('express');
const { detectTopBottom } = require('../helpers/imageProcessing');
const router = express.Router();
const path = require('path');

// Detect top and bottom from uploaded image
router.post('/', async (req, res) => {
  const { imagePath } = req.body; // Assume frontend sends the uploaded file path

  if (!imagePath) {
    return res.status(400).json({ error: 'Image path is required' });
  }

  try {
    const detectionResults = await detectTopBottom(path.join(__dirname, '../uploads', imagePath));
    res.status(200).json(detectionResults);
  } catch (error) {
    res.status(500).json({ error: 'Error detecting top and bottom' });
  }
});

module.exports = router;