import React from 'react';
import { NavLink } from 'react-router-dom';

/**
 * 區段內子導覽（/breadth/*、/conviction/*、/legacy/* 共用）。
 *
 * item.icon 收 React 元素（lucide 圖示），不再用 emoji：
 * emoji 在不同作業系統長得不一樣，也吃不到色彩 token。
 * 窄螢幕會把文字標籤藏起來只剩圖示，所以 aria-label 一定要帶上，
 * 不然螢幕閱讀器與觸控使用者就只看得到一個沒有名字的按鈕。
 */
export default function SectionSubNav({ items }) {
  return (
    <nav
      aria-label="子頁導覽"
      style={{
        display: 'flex', gap: 'var(--space-1)', flexWrap: 'wrap',
        marginBottom: 'var(--space-5)',
        paddingBottom: 'var(--space-3)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      {items.map(item => (
        <NavLink
          key={item.to} to={item.to} end={item.end}
          aria-label={item.label}
          title={item.hint || item.label}
          className={({ isActive }) => `nav-tab${isActive ? ' active' : ''}`}
        >
          <span className="tab-icon" aria-hidden="true">{item.icon}</span>
          <span>{item.label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
