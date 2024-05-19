import { useNavigate, useParams } from "react-router-dom";
import { useCallback } from "react";

export function useRoutedApp () {
  let { chatId } = useParams<{ chatId?: string }>();
  if (chatId === '' || chatId === 'new') chatId = undefined
  const navigate = useNavigate();

  const goToChat = useCallback((chatId: string = 'new') => {
    navigate(`/chats/${chatId}`)
  }, [navigate])

  return { chatId, goToChat }
}
