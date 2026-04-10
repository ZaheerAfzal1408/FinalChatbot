import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Info, Wifi, AlertTriangle, CheckCircle2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE = 'http://localhost:8000';

const ResultCards = ({ data }) => (
    <div className="result-cards">
        {data.map((item, idx) => {
            const isAnomaly = item.anomaly === 1;
            return (
                <div key={idx} className={`result-card ${isAnomaly ? 'anomaly' : 'normal'}`}>
                    <div className="rc-head">
                        <span className="rc-name">{item.asset || 'Industrial Asset'}</span>
                        <span className={`rc-pill ${isAnomaly ? 'pill-red' : 'pill-green'}`}>
                            {isAnomaly ? <AlertTriangle size={10} /> : <CheckCircle2 size={10} />}
                            {isAnomaly ? 'Anomaly Detected' : 'Nominal Status'}
                        </span>
                    </div>
                    <div className="rc-metrics">
                        <span>MSE: <strong>{item.mse?.toFixed(5) || '0.00000'}</strong></span>
                        <span>Threshold: <strong>{item.threshold?.toFixed(5) || '0.00000'}</strong></span>
                    </div>
                </div>
            );
        })}
    </div>
);

const MessageBubble = ({ message }) => {
    const isAI = message.sender === 'ai';
    const results = message.result
        ? (message.result.multi_report
            ? message.result.multi_report
            : Array.isArray(message.result)
                ? message.result
                : [message.result])
        : null;

    return (
        <motion.div
            initial={{ y: 16, opacity: 0, scale: 0.97 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className={`msg-row ${message.sender}`}
        >
            <div className={`bubble-avatar ${message.sender}`}>
                {isAI ? <Bot size={14} /> : <User size={14} />}
            </div>
            <div className={`bubble ${message.sender}`}>
                {isAI ? (
                    <div className="markdown-content">
                        <ReactMarkdown>{message.text}</ReactMarkdown>
                    </div>
                ) : (
                    <p>{message.text}</p>
                )}
            </div>
        </motion.div>
    );
};

const ChatPanel = () => {
    const [messages, setMessages] = useState([
        {
            id: 1,
            text: "Hello! I am **The Foreman**, your Industrial Multi-Agent Supervisor. All system components are online and reporting nominal. How can I assist you today?",
            sender: 'ai'
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, loading]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), text: input, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const { data } = await axios.post(`${API_BASE}/api/chat`, { message: input });
            const aiMsg = {
                id: Date.now() + 1,
                text: data.reply,
                sender: 'ai',
                result: data.data,
                insight: data.insight
            };
            setMessages(prev => [...prev, aiMsg]);
        } catch {
            setMessages(prev => [...prev, {
                id: Date.now(),
                text: "The Foreman is currently unable to reach the gateway. Please check your network connection.",
                sender: 'ai'
            }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="foreman-chat">
            {/* Topbar */}
            <div className="chat-topbar">
                <div className="topbar-avatar">
                    <Bot size={18} />
                </div>
                <div className="topbar-meta">
                    <h2 className="topbar-title">The Foreman</h2>
                    <span className="topbar-sub">Supervising Industrial Systems</span>
                </div>
                <div className="topbar-status">
                    <Wifi size={12} />
                    All systems reporting
                </div>
            </div>

            {/* Messages */}
            <div className="chat-messages" ref={scrollRef}>

                {/* Guide Card */}
                <div className="guide-card">
                    <div className="guide-icon"><Info size={18} /></div>
                    <div className="guide-body">
                        <p className="guide-title">Operational Guide</p>
                        <p className="guide-desc">To analyze specific systems, use these naming conventions:</p>
                        <ul className="guide-list">
                            <li><strong>Coldrooms:</strong> "coldroom 1" through "coldroom 9"</li>
                            <li><strong>Refinery:</strong> "tank 1" to "tank 13", no tank 10</li>
                        </ul>
                    </div>
                </div>

                <AnimatePresence>
                    {messages.map((m) => (
                        <MessageBubble key={m.id} message={m} />
                    ))}
                </AnimatePresence>

                {loading && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="typing-row"
                    >
                        <div className="bubble-avatar ai"><Bot size={14} /></div>
                        <div className="typing-dots">
                            <span /><span /><span />
                        </div>
                    </motion.div>
                )}
            </div>


            {/* Input */}
            <div className="chat-input-area">
                <form className="input-form" onSubmit={handleSend}>
                    <input
                        type="text"
                        className="input-field"
                        placeholder="Type a message to The Foreman..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                    />
                    <button type="submit" className="send-btn" disabled={loading}>
                        <Send size={15} />
                    </button>
                </form>
            </div>
        </main>
    );
};


export default ChatPanel;