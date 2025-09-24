import React, { useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import LoginPage from './components/LoginPage'
import SignupPage from './components/SignupPage'
import ImageUpload from './components/ImageUpload'
import './App.css'

const AppContent: React.FC = () => {
  const { user, loading, signOut } = useAuth()
  const [isLogin, setIsLogin] = useState(true)

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return isLogin ? (
      <LoginPage onSwitchToSignup={() => setIsLogin(false)} />
    ) : (
      <SignupPage onSwitchToLogin={() => setIsLogin(true)} />
    )
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">Vortex AI</h1>
          <div className="user-info">
            <span className="user-name">Welcome, {user.name}</span>
            <button 
              onClick={signOut} 
              className="logout-btn"
            >
              Logout
            </button>
          </div>
        </div>
      </header>
      <main className="app-main">
        <ImageUpload />
      </main>
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
