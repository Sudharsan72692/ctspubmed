import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import Sidebar from './components/Sidebar';
import Home from './components/Home';
import SearchPage from './components/SearchPage';
import Profile from './components/Profile';
import History from './components/History';
import Header from './components/Header';
import GroqSearchPage from './components/GroqSearchPage';
import GroqHistory from './components/GroqHistory';
import Dashboard from './components/Dashboard';
import ChatbotPage from './components/ChatbotPage';

const ProtectedLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isLoggedIn = !!localStorage.getItem('loggedInUser');
  if (!isLoggedIn) return <Navigate to="/login" />;
  return (
    <div className="min-h-screen font-sans flex flex-col">
      <Header onFilterToggle={() => {}} />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="ml-56 flex-1 bg-slate-50 min-h-0 overflow-auto">{children}</main>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedLayout>
              <Dashboard />
            </ProtectedLayout>
          }
        />
        <Route
          path="/home"
          element={
            <ProtectedLayout>
              <Home />
            </ProtectedLayout>
          }
        />
        <Route
          path="/search"
          element={
            <ProtectedLayout>
              <SearchPage />
            </ProtectedLayout>
          }
        />
        <Route
          path="/groq"
          element={
            <ProtectedLayout>
              <GroqSearchPage />
            </ProtectedLayout>
          }
        />
        <Route
          path="/history"
          element={
            <ProtectedLayout>
              <History />
            </ProtectedLayout>
          }
        />
        <Route
          path="/history/groq"
          element={
            <ProtectedLayout>
              <GroqHistory />
            </ProtectedLayout>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedLayout>
              <Profile />
            </ProtectedLayout>
          }
        />
        <Route
          path="/chatbot"
          element={
            <ProtectedLayout>
              <ChatbotPage />
            </ProtectedLayout>
          }
        />
        <Route
  path="/"
  element={
    localStorage.getItem("loggedInUser")
      ? <Navigate to="/home" />
      : <Navigate to="/login" />
  }
/>
        <Route path="*" element={<Navigate to="/home" />} />
      </Routes>
    </Router>
  );
};

export default App;
