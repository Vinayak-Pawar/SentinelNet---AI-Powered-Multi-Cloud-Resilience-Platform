#!/usr/bin/env python3
"""
SentinelNet Communication Manager
Handles multi-channel communication for distributed agents

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module implements the communication layer with:
- HTTP/WebSocket for cloud coordination
- WebRTC for P2P mesh networking
- SMS/Email alerts for internet outages
- Local network broadcast as last resort
- Automatic failover between channels
"""

import asyncio
import logging
import json
import websockets
import aiohttp
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
import socket
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)

class CommunicationChannel(Enum):
    """Enumeration of available communication channels"""
    HTTP_WEBSOCKET = "http_websocket"
    WEBRTC_P2P = "webrtc_p2p"
    SMS = "sms"
    EMAIL = "email"
    LOCAL_BROADCAST = "local_broadcast"

class CommunicationStatus(Enum):
    """Status of communication channels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"

@dataclass
class CommunicationConfig:
    """Configuration for communication channels"""
    # HTTP/WebSocket settings
    websocket_port: int = 8080
    http_timeout: int = 30

    # WebRTC settings
    webrtc_enabled: bool = True
    stun_servers: List[str] = None

    # SMS settings (Twilio)
    sms_enabled: bool = False
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

    # Email settings
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_recipients: List[str] = None

    # Local broadcast settings
    broadcast_port: int = 9000
    broadcast_enabled: bool = True

    def __post_init__(self):
        if self.stun_servers is None:
            self.stun_servers = [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ]
        if self.alert_email_recipients is None:
            self.alert_email_recipients = []

@dataclass
class Message:
    """Data class for inter-agent messages"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcasts
    message_type: str  # "heartbeat", "alert", "investigation_request", "notification"
    payload: Dict[str, Any]
    timestamp: datetime
    priority: str = "normal"  # "low", "normal", "high", "critical"
    ttl_seconds: int = 300  # Time to live

@dataclass
class ChannelStatus:
    """Status information for a communication channel"""
    channel: CommunicationChannel
    status: CommunicationStatus
    last_successful_use: Optional[datetime] = None
    error_count: int = 0
    latency_ms: Optional[int] = None

class CommunicationManager:
    """
    Main communication manager for SentinelNet
    Handles all inter-agent communication with automatic failover
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        """
        Initialize the communication manager

        Args:
            config: Communication configuration
        """
        self.config = config or CommunicationConfig()

        # Channel status tracking
        self.channel_status: Dict[CommunicationChannel, ChannelStatus] = {}
        for channel in CommunicationChannel:
            self.channel_status[channel] = ChannelStatus(
                channel=channel,
                status=CommunicationStatus.UNKNOWN
            )

        # WebSocket connections (agent_id -> websocket)
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.websocket_server: Optional[websockets.WebSocketServer] = None

        # HTTP client session
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Local broadcast socket
        self.broadcast_socket: Optional[socket.socket] = None
        self.broadcast_thread: Optional[threading.Thread] = None

        # Message queues and handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: asyncio.Queue = asyncio.Queue()
        self.message_history: List[Message] = []

        # Agent discovery
        self.discovered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_heartbeat_timeout = 60  # seconds

        # Initialize components
        self._setup_default_handlers()

        logger.info("📡 Communication Manager initialized")

    def _setup_default_handlers(self):
        """Set up default message handlers"""
        self.message_handlers = {
            "heartbeat": self._handle_heartbeat,
            "alert": self._handle_alert,
            "investigation_request": self._handle_investigation_request,
            "investigation_response": self._handle_investigation_response,
            "notification": self._handle_notification,
            "agent_discovery": self._handle_agent_discovery
        }

    async def initialize(self):
        """Initialize all communication channels"""
        logger.info("🔄 Initializing communication channels...")

        # Initialize HTTP client
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.http_timeout)
        )

        # Start WebSocket server
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "0.0.0.0",
                self.config.websocket_port,
                ping_interval=30,
                ping_timeout=10
            )
            self.channel_status[CommunicationChannel.HTTP_WEBSOCKET].status = CommunicationStatus.HEALTHY
            logger.info(f"✅ WebSocket server started on port {self.config.websocket_port}")
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            self.channel_status[CommunicationChannel.HTTP_WEBSOCKET].status = CommunicationStatus.DOWN

        # Start local broadcast listener
        if self.config.broadcast_enabled:
            try:
                self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.broadcast_socket.bind(("", self.config.broadcast_port))

                self.broadcast_thread = threading.Thread(target=self._broadcast_listener, daemon=True)
                self.broadcast_thread.start()

                self.channel_status[CommunicationChannel.LOCAL_BROADCAST].status = CommunicationStatus.HEALTHY
                logger.info(f"✅ Local broadcast listener started on port {self.config.broadcast_port}")
            except Exception as e:
                logger.error(f"❌ Failed to start broadcast listener: {e}")
                self.channel_status[CommunicationChannel.LOCAL_BROADCAST].status = CommunicationStatus.DOWN

        # Test channel connectivity
        await self._test_channel_connectivity()

        logger.info("✅ Communication channels initialized")

    async def close(self):
        """Close all communication channels"""
        logger.info("🔄 Closing communication channels...")

        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

        # Close HTTP session
        if self.http_session:
            await self.http_session.close()

        # Close broadcast socket
        if self.broadcast_socket:
            self.broadcast_socket.close()

        # Clear connections
        self.websocket_connections.clear()

        logger.info("✅ Communication channels closed")

    async def get_status(self) -> str:
        """
        Get overall communication status

        Returns:
            Status string
        """
        healthy_channels = sum(
            1 for status in self.channel_status.values()
            if status.status == CommunicationStatus.HEALTHY
        )

        total_channels = len(self.channel_status)

        if healthy_channels == total_channels:
            return "healthy"
        elif healthy_channels >= total_channels // 2:
            return "degraded"
        else:
            return "critical"

    async def send_message(self, message: Message,
                          preferred_channel: CommunicationChannel = None) -> bool:
        """
        Send a message using the best available channel

        Args:
            message: Message to send
            preferred_channel: Preferred communication channel

        Returns:
            bool: Success status
        """
        channels_to_try = self._get_channel_priority(preferred_channel)

        for channel in channels_to_try:
            try:
                success = await self._send_via_channel(message, channel)
                if success:
                    self.channel_status[channel].last_successful_use = datetime.now()
                    self.channel_status[channel].error_count = 0
                    self.channel_status[channel].status = CommunicationStatus.HEALTHY
                    return True
                else:
                    self.channel_status[channel].error_count += 1
                    if self.channel_status[channel].error_count > 3:
                        self.channel_status[channel].status = CommunicationStatus.DEGRADED

            except Exception as e:
                logger.warning(f"⚠️ Failed to send via {channel.value}: {e}")
                self.channel_status[channel].error_count += 1

        logger.error(f"❌ Failed to send message {message.message_id} via any channel")
        return False

    def _get_channel_priority(self, preferred: Optional[CommunicationChannel] = None) -> List[CommunicationChannel]:
        """Get channel priority order"""
        if preferred:
            channels = [preferred]
        else:
            channels = []

        # Add remaining channels in priority order
        priority_order = [
            CommunicationChannel.HTTP_WEBSOCKET,
            CommunicationChannel.WEBRTC_P2P,
            CommunicationChannel.LOCAL_BROADCAST,
            CommunicationChannel.EMAIL,
            CommunicationChannel.SMS
        ]

        for channel in priority_order:
            if channel not in channels:
                channels.append(channel)

        return channels

    async def _send_via_channel(self, message: Message, channel: CommunicationChannel) -> bool:
        """Send message via specific channel"""
        if channel == CommunicationChannel.HTTP_WEBSOCKET:
            return await self._send_websocket(message)
        elif channel == CommunicationChannel.WEBRTC_P2P:
            return await self._send_webrtc(message)
        elif channel == CommunicationChannel.LOCAL_BROADCAST:
            return await self._send_broadcast(message)
        elif channel == CommunicationChannel.EMAIL:
            return await self._send_email(message)
        elif channel == CommunicationChannel.SMS:
            return await self._send_sms(message)
        else:
            return False

    async def _send_websocket(self, message: Message) -> bool:
        """Send message via WebSocket"""
        if not self.websocket_connections:
            return False

        message_data = json.dumps(asdict(message), default=str)

        if message.recipient_id and message.recipient_id in self.websocket_connections:
            # Direct message
            try:
                await self.websocket_connections[message.recipient_id].send(message_data)
                return True
            except Exception as e:
                logger.warning(f"⚠️ WebSocket send failed to {message.recipient_id}: {e}")
                return False
        else:
            # Broadcast to all connected agents
            success_count = 0
            for agent_id, ws in self.websocket_connections.items():
                try:
                    await ws.send(message_data)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ WebSocket broadcast failed to {agent_id}: {e}")

            return success_count > 0

    async def _send_webrtc(self, message: Message) -> bool:
        """Send message via WebRTC P2P (placeholder for future implementation)"""
        # WebRTC implementation would go here
        # For now, fall back to WebSocket
        logger.warning("⚠️ WebRTC not fully implemented, using WebSocket fallback")
        return await self._send_websocket(message)

    async def _send_broadcast(self, message: Message) -> bool:
        """Send message via local network broadcast"""
        if not self.broadcast_socket:
            return False

        try:
            message_data = json.dumps(asdict(message), default=str)
            self.broadcast_socket.sendto(
                message_data.encode(),
                ("<broadcast>", self.config.broadcast_port)
            )
            return True
        except Exception as e:
            logger.error(f"❌ Broadcast send failed: {e}")
            return False

    async def _send_email(self, message: Message) -> bool:
        """Send message via email"""
        if not self.config.email_enabled or not self.config.smtp_username:
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.alert_email_recipients)
            msg['Subject'] = f"SentinelNet Alert: {message.message_type.upper()}"

            body = f"""
SentinelNet Alert

Type: {message.message_type}
Sender: {message.sender_id}
Priority: {message.priority}
Timestamp: {message.timestamp}

Details:
{json.dumps(message.payload, indent=2)}

This is an automated message from SentinelNet.
            """
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"❌ Email send failed: {e}")
            return False

    async def _send_sms(self, message: Message) -> bool:
        """Send message via SMS (placeholder for Twilio integration)"""
        if not self.config.sms_enabled:
            return False

        # Twilio SMS implementation would go here
        logger.warning("⚠️ SMS not implemented yet")
        return False

    async def _handle_websocket_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connections"""
        try:
            # Extract agent ID from path or initial message
            agent_id = None

            async for message_str in websocket:
                try:
                    message_data = json.loads(message_str)
                    message = Message(**message_data)

                    # Register agent if not already known
                    if not agent_id:
                        agent_id = message.sender_id
                        self.websocket_connections[agent_id] = websocket
                        logger.info(f"✅ Agent {agent_id} connected via WebSocket")

                    # Handle the message
                    await self._handle_incoming_message(message)

                except json.JSONDecodeError:
                    logger.warning("⚠️ Received invalid JSON message")
                except Exception as e:
                    logger.error(f"❌ Error handling WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            # Agent disconnected
            if agent_id and agent_id in self.websocket_connections:
                del self.websocket_connections[agent_id]
                logger.info(f"🔌 Agent {agent_id} disconnected")

    def _broadcast_listener(self):
        """Listen for local broadcast messages"""
        while self.broadcast_socket:
            try:
                data, addr = self.broadcast_socket.recvfrom(4096)
                message_str = data.decode()

                try:
                    message_data = json.loads(message_str)
                    message = Message(**message_data)

                    # Handle the message in a new task
                    asyncio.create_task(self._handle_incoming_message(message))

                except json.JSONDecodeError:
                    logger.warning("⚠️ Received invalid broadcast message")
                except Exception as e:
                    logger.error(f"❌ Error handling broadcast message: {e}")

            except OSError:
                # Socket closed
                break
            except Exception as e:
                logger.error(f"❌ Broadcast listener error: {e}")

    async def _handle_incoming_message(self, message: Message):
        """Handle incoming messages from any channel"""
        # Add to message history
        self.message_history.append(message)
        if len(self.message_history) > 1000:  # Keep last 1000 messages
            self.message_history = self.message_history[-1000:]

        # Update agent discovery
        if message.sender_id not in self.discovered_agents:
            self.discovered_agents[message.sender_id] = {
                'last_seen': message.timestamp,
                'message_count': 0
            }
        self.discovered_agents[message.sender_id]['last_seen'] = message.timestamp
        self.discovered_agents[message.sender_id]['message_count'] += 1

        # Route to appropriate handler
        if message.message_type in self.message_handlers:
            try:
                await self.message_handlers[message.message_type](message)
            except Exception as e:
                logger.error(f"❌ Error handling {message.message_type}: {e}")
        else:
            logger.warning(f"⚠️ No handler for message type: {message.message_type}")

    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages"""
        # Update agent status
        agent_info = message.payload.get('agent_info', {})
        self.discovered_agents[message.sender_id].update({
            'status': 'healthy',
            'last_heartbeat': message.timestamp,
            'services': agent_info.get('services', [])
        })

    async def _handle_alert(self, message: Message):
        """Handle alert messages"""
        alert_data = message.payload
        logger.warning(f"🚨 Alert from {message.sender_id}: {alert_data}")

        # Forward to orchestrator for processing
        await self.pending_messages.put(message)

    async def _handle_investigation_request(self, message: Message):
        """Handle investigation requests"""
        logger.info(f"🔍 Investigation request from {message.sender_id}")

        # This would trigger local investigation
        await self.pending_messages.put(message)

    async def _handle_investigation_response(self, message: Message):
        """Handle investigation responses"""
        logger.info(f"📋 Investigation response from {message.sender_id}")

        # Forward to requesting agent
        await self.pending_messages.put(message)

    async def _handle_notification(self, message: Message):
        """Handle notification messages"""
        notification_data = message.payload
        logger.info(f"📢 Notification from {message.sender_id}: {notification_data}")

    async def _handle_agent_discovery(self, message: Message):
        """Handle agent discovery messages"""
        agent_info = message.payload
        self.discovered_agents[message.sender_id] = {
            **self.discovered_agents.get(message.sender_id, {}),
            **agent_info,
            'last_seen': message.timestamp
        }
        logger.info(f"🔍 Agent discovered: {message.sender_id}")

    async def request_investigation(self, agent_id: str, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request investigation from a specific agent

        Args:
            agent_id: Target agent ID
            incident_data: Incident information

        Returns:
            Investigation results
        """
        message = Message(
            message_id=f"investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id="orchestrator",
            recipient_id=agent_id,
            message_type="investigation_request",
            payload=incident_data,
            timestamp=datetime.now(),
            priority="high"
        )

        # Send request
        success = await self.send_message(message)
        if not success:
            return {"error": "Failed to send investigation request"}

        # Wait for response (simplified - in real implementation would use proper async waiting)
        await asyncio.sleep(5)  # Wait for response

        # Check for response in pending messages
        response_data = {"status": "investigation_sent", "agent_contacted": agent_id}
        return response_data

    async def send_notification(self, notification_data: Dict[str, Any]):
        """
        Send notification via available channels

        Args:
            notification_data: Notification content
        """
        message = Message(
            message_id=f"notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id="orchestrator",
            recipient_id=None,  # Broadcast
            message_type="notification",
            payload=notification_data,
            timestamp=datetime.now(),
            priority="high"
        )

        await self.send_message(message)

    async def _test_channel_connectivity(self):
        """Test connectivity of all channels"""
        logger.info("🔍 Testing channel connectivity...")

        # Test HTTP connectivity
        try:
            async with self.http_session.get("https://httpbin.org/get", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    self.channel_status[CommunicationChannel.HTTP_WEBSOCKET].status = CommunicationStatus.HEALTHY
                else:
                    self.channel_status[CommunicationChannel.HTTP_WEBSOCKET].status = CommunicationStatus.DEGRADED
        except Exception:
            self.channel_status[CommunicationChannel.HTTP_WEBSOCKET].status = CommunicationStatus.DEGRADED

        # Test email connectivity (if enabled)
        if self.config.email_enabled:
            # Simple SMTP connection test
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.config.smtp_username, self.config.smtp_password)
                self.channel_status[CommunicationChannel.EMAIL].status = CommunicationStatus.HEALTHY
            except Exception:
                self.channel_status[CommunicationChannel.EMAIL].status = CommunicationStatus.DOWN

        logger.info("✅ Channel connectivity test complete")

    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all communication channels

        Returns:
            Dictionary with channel status information
        """
        return {
            channel.value: {
                'status': status.status.value,
                'last_successful_use': status.last_successful_use.isoformat() if status.last_successful_use else None,
                'error_count': status.error_count,
                'latency_ms': status.latency_ms
            }
            for channel, status in self.channel_status.items()
        }

    def get_discovered_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about discovered agents

        Returns:
            Dictionary with agent information
        """
        # Clean up old agents
        current_time = datetime.now()
        active_agents = {}

        for agent_id, info in self.discovered_agents.items():
            last_seen = info.get('last_seen', datetime.min)
            if (current_time - last_seen).total_seconds() < self.agent_heartbeat_timeout * 3:
                active_agents[agent_id] = info

        self.discovered_agents = active_agents
        return active_agents

# Global communication manager instance
_communication_manager_instance: Optional[CommunicationManager] = None

def get_communication_manager() -> CommunicationManager:
    """Get the global communication manager instance"""
    global _communication_manager_instance
    if _communication_manager_instance is None:
        _communication_manager_instance = CommunicationManager()
    return _communication_manager_instance

# Convenience functions
async def initialize_communication() -> CommunicationManager:
    """Initialize and return the communication manager"""
    manager = get_communication_manager()
    await manager.initialize()
    return manager

async def send_message_to_agent(agent_id: str, message_type: str, payload: Dict[str, Any]) -> bool:
    """Send a message to a specific agent"""
    manager = get_communication_manager()
    message = Message(
        message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        sender_id="system",
        recipient_id=agent_id,
        message_type=message_type,
        payload=payload,
        timestamp=datetime.now()
    )
    return await manager.send_message(message)

if __name__ == "__main__":
    # Test the communication manager
    async def test_communication():
        manager = CommunicationManager()
        await manager.initialize()

        # Test status
        status = await manager.get_status()
        print(f"Communication status: {status}")

        # Test channel status
        channels = manager.get_channel_status()
        print("Channel status:", json.dumps(channels, indent=2))

        await manager.close()

    asyncio.run(test_communication())
