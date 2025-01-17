import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';
import '@testing-library/jest-dom';

describe('App', () => {
    it('renders PlanAid heading', () => {
        render(<App />);
        const headingElements = screen.getAllByText(/PlanAid/i);
        expect(headingElements.length).toBeGreaterThan(0); // Ensure at least one match
    });
});
