import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FileUpload from '../components/FileUpload';

describe('FileUpload', () => {
  test('renders file input', () => {
    render(<FileUpload />);
    const fileInput = screen.getByRole('button');
    expect(fileInput).toBeInTheDocument();
  });

  // Add more tests as needed
});