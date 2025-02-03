const { exec } = require('child_process');
const path = require('path');

// Function to call the Python Faster R-CNN script
const detectTopBottom = async (imagePath) => {
  const scriptPath = path.join(__dirname, '../model/detect.py'); // Path to Python script

  return new Promise((resolve, reject) => {
    exec(`python3 ${scriptPath} ${imagePath}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${stderr}`);
        reject(`Error processing image: ${stderr}`);
      } else {
        const detectionResults = JSON.parse(stdout); // Parse JSON results
        resolve(detectionResults);
      }
    });
  });
};

module.exports = { detectTopBottom };

