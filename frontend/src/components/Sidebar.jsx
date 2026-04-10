import React from 'react';
import { Shield, Server, Database, Thermometer, Droplets, AlertTriangle, CheckCircle2 } from 'lucide-react';

const Sidebar = () => {
    return (
        <aside className="foreman-sidebar">
            {/* Brand */}
            <div className="sb-brand">
                <div className="sb-logo">
                    <Shield size={20} fill="white" stroke="none" />
                </div>
                <div>
                    <h2 className="sb-title">The Foreman</h2>
                    <div className="sb-online">
                        <span className="dot-pulse" />
                        System Online
                    </div>
                </div>
            </div>

            {/* Network Status */}
            <div className="sb-section">
                <p className="sb-label">Network Status</p>
                <div className="net-item">
                    <span className="net-left"><Server size={13} /> Gateway API</span>
                    <span className="pill pill-green">Connected</span>
                </div>
                <div className="net-item">
                    <span className="net-left"><Database size={13} /> PostgreSQL</span>
                    <span className="pill pill-green">Connected</span>
                </div>
            </div>

            <div className="sb-divider" />

            {/* Monitored Nodes */}
            <div className="sb-section">
                <p className="sb-label">Monitored Nodes</p>

                <div className="node-card">
                    <div className="nc-head">
                        <span className="nc-name"><Thermometer size={14} /> Coldrooms</span>
                        <span className="pill pill-active">10 Active</span>
                    </div>
                </div>

                <div className="node-card">
                    <div className="nc-head">
                        <span className="nc-name"><Droplets size={14} /> Refinery</span>
                        <span className="pill pill-active">12 Active</span>
                    </div>
                    <p className="nc-sub">Tanks T1–T13 &nbsp;·&nbsp; <span className="text-warn">T10 unavailable</span></p>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;