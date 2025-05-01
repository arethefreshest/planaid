import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FileUpload from '../components/FileUpload';
import axios from 'axios';
import '@testing-library/jest-dom';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('FileUpload Component', () => {
    it('renders and uploads a file', async () => {
        mockedAxios.post.mockResolvedValue({ data: { text: 'Test PDF Content' } });

        const mockOnUploadSuccess = jest.fn();

        render(<FileUpload onUploadSuccess={mockOnUploadSuccess} />);

        const fileInput = screen.getByTestId('file-input');
        const uploadButton = screen.getByText('Upload');

        fireEvent.change(fileInput, {
            target: { files: [new File(['dummy content'], 'test.pdf')] },
        });

        fireEvent.click(uploadButton);

        const result = await screen.findByText('Test PDF Content');
        expect(result).toBeInTheDocument();
        
        expect(mockOnUploadSuccess).toHaveBeenCalledWith({ text: 'Test PDF Content' });
    });
});