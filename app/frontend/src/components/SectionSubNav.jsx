import React from 'react';
import { NavLink } from 'react-router-dom';

// 區段內子導覽（/conviction/* 與 /breadth/* 共用）
export default function SectionSubNav({ items }) {
  return (
    <div style={{
      display: 'flex', gap: 4, flexWrap: 'wrap',
      marginBottom: 20, paddingBottom: 12, borderBottom: '1px solid var(--border)',
    }}>
      {items.map(item => (
        <NavLink
          key={item.to} to={item.to} end={item.end}
          className={({ isActive }) => `nav-tab${isActive ? ' active' : ''}`}
        >
          <span className="tab-icon">{item.icon}</span>
          <span>{item.label}</span>
        </NavLink>
      ))}
    </div>
  );
}
