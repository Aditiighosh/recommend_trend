const express = require("express");
const multer = require("multer");
const router = express.Router();

const upload = multer({ dest: "uploads/" }); // Temp storage for images

router.post("/", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  // Process file here or pass to Flask
  res.json({ message: "File uploaded successfully", file: req.file });
});

module.exports = router;
