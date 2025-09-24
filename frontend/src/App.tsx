import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import ImageUpload from './components/ImageUpload'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <ImageUpload />
    </>
  )
}

export default App
