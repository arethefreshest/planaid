import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';
import { describe, it, expect } from '@jest/globals';
import '@testing-library/jest-dom';

describe('App Component', () => {
    it('renders PlanAid heading', () => {
        render(<App />);
        const headingElements = screen.getAllByText(/PlanAid/i);
        expect(headingElements.length).toBeGreaterThan(0);
    });
});
