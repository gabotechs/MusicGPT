import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import App from "./App.tsx";

function RoutedApp () {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={'/'} element={<Navigate to="/chats/new"/>}/>
        <Route path={'/chats'} element={<Navigate to="/chats/new"/>}/>
        <Route path={'/chats/:chatId'} element={<App/>}/>
      </Routes>
    </BrowserRouter>
  )
}

export default RoutedApp;
