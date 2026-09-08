import { useEffect, useRef, useState } from "react";

// 订阅后端事件流（/api/events）：onEvent 收 {type, data}；返回连接状态。
// dev 走 Vite 代理（ws 转发），生产同源直连；断线 2s 自动重连。
export function useEvents(onEvent) {
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${window.location.host}/api/events`;
    let ws = null;
    let closed = false;
    let retry = null;

    const connect = () => {
      ws = new WebSocket(url);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (handlerRef.current) handlerRef.current(msg);
        } catch {
          /* 忽略非法帧 */
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 2000);
      };
    };
    connect();

    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      if (ws) ws.close();
    };
  }, []);

  return connected;
}
