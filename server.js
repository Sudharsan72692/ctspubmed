// server.js
import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import bodyParser from "body-parser";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

const app = express();
app.use(cors());
app.use(bodyParser.json());

// -----------------------------
// MongoDB Connection
// -----------------------------
const mongoURI =
  "mongodb+srv://praneeshroshan_db_user:0F6f0m54x4MJ9Qbz@cts.aevkhjk.mongodb.net/?retryWrites=true&w=majority&appName=CTS";

mongoose
  .connect(mongoURI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    dbName: "pubmed_db",
  })
  .then(() => console.log("✅ Connected to MongoDB (Cosmos DB)"))
  .catch((err) => console.error("❌ MongoDB connection error:", err));

// -----------------------------
// User Schema & Model
// -----------------------------
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  jwt_token: { type: String }, // <-- store JWT here
});

const User = mongoose.model("User", userSchema);

// -----------------------------
// JWT Secret
// -----------------------------
const JWT_SECRET = "supersecretkey";

// -----------------------------
// Health check route
// -----------------------------
app.get("/", (req, res) => {
  res.send("API running ✅");
});

// -----------------------------
// Register Endpoint
// -----------------------------
app.post("/register", async (req, res) => {
  const { name, email, password } = req.body;
  try {
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ name, email, password: hashedPassword });
    await newUser.save();

    res.json({ message: "Registration successful" });
  } catch (err) {
    console.error("Registration error:", err);
    res.status(500).json({ error: "Server error during registration" });
  }
});

// -----------------------------
// Login Endpoint (store JWT in DB)
// -----------------------------
app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ error: "Invalid email or password" });

    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(400).json({ error: "Invalid email or password" });

    // Generate JWT
    const token = jwt.sign({ id: user._id, email: user.email }, JWT_SECRET, { expiresIn: "2h" });

    // Store JWT in the user document
    user.jwt_token = token;
    await user.save();

    res.json({
      message: "Login successful",
      token, // optional to return to frontend
      user: { id: user._id, name: user.name, email: user.email },
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Server error during login" });
  }
});

// -----------------------------
// Retrieve JWT directly from DB (example route)
// -----------------------------
app.get("/get-token/:email", async (req, res) => {
  const { email } = req.params;
  try {
    const user = await User.findOne({ email });
    if (!user || !user.jwt_token) return res.status(404).json({ error: "Token not found" });
    res.json({ token: user.jwt_token });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Server error" });
  }
});

// -----------------------------
// Protected Route Example
// -----------------------------
app.get("/profile", async (req, res) => {
  const authHeader = req.headers["authorization"];
  if (!authHeader) return res.status(401).json({ error: "Missing token" });

  const token = authHeader.split(" ")[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const user = await User.findById(decoded.id).select("-password");
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json(user);
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
});

// -----------------------------
// Chatbot API Proxy
// -----------------------------
app.post('/api/chatbot/:mode', async (req, res) => {
  const { mode } = req.params;
  const { user_input } = req.body;
  
  try {
    const validModes = ['concept', 'literature_review', 'citation', 'exam_notes'];
    if (!validModes.includes(mode)) {
      return res.status(400).json({ error: 'Invalid mode' });
    }
    
    // Forward request to the Python FastAPI backend
    const response = await fetch(`http://localhost:8001/${mode}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_input }),
    });
    
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Chatbot API error:', err);
    res.status(500).json({ error: 'Error connecting to chatbot service' });
  }
});

// -----------------------------
// Start Server
// -----------------------------
app.listen(5000, () => console.log("🚀 Server running on port 5000"));
