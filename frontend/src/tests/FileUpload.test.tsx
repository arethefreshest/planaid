import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import FileUpload from '../components/FileUpload';
import axios from 'axios';
import '@testing-library/jest-dom';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('FileUpload Component', () => {
    it('renders and uploads a file', async () => {
        mockedAxios.post.mockResolvedValueOnce({
            data: {
                title: 'Test Document',
                totalPages: 1,
                processedDate: '2024-02-04T12:00:00Z',
                pages: [
                    {
                        pageNumber: 1,
                        content: 'Test PDF Content'
                    }
                ]
            }
        });

        render(<FileUpload />);

        const file = new File(['test'], 'test.pdf', { type: 'application/pdf' });

        const input = screen.getByTestId('file-input');
        fireEvent.change(input, { target: { files: [file] } });

        const uploadButton = screen.getByText('Upload');
        fireEvent.click(uploadButton);

        await waitFor(() => {
            expect(screen.getByText('Test Document')).toBeInTheDocument();
        });

        expect(mockedAxios.post).toHaveBeenCalledWith(
            '/api/validation',
            expect.any(FormData),
            expect.any(Object)
        );
    });
});