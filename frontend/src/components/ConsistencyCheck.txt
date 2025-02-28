/**
 * Consistency Check Component
 * 
 * Main component for handling regulatory document consistency checks.
 * Allows users to upload and compare plankart, bestemmelser, and SOSI files.
 * 
 * Features:
 * - File upload handling
 * - Progress tracking
 * - Error handling
 * - Results display
 * - Metadata visualization
 */

import React, { useState } from 'react';
import axios from 'axios';
import { CircularProgress } from './CircularProgress';
import { logger } from '../utils/logger';
import type { ConsistencyResult } from '../types';

// Configuration constants
const ALLOWED_TYPES = ['application/pdf', 'text/xml'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

/**
 * Validates uploaded files for type and size constraints
 */
const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
        return 'Ugyldig filformat. Vennligst last opp PDF eller XML fil.';
    }
    if (file.size > MAX_FILE_SIZE) {
        return 'Filen er for stor. Maksimal størrelse er 10MB.';
    }
    return null;
};

// Add logging interceptors
axios.interceptors.request.use(request => {
  logger.info('Starting Request:', {
    url: request.url,
    method: request.method,
    data: request.data instanceof FormData ? 'FormData' : request.data
  });
  return request;
});

axios.interceptors.response.use(
  response => {
    logger.info('Response:', {
      status: response.status,
      url: response.config.url
    });
    return response;
  },
  error => {
    logger.error('Request Error:', {
      message: error.message,
      url: error.config?.url,
      status: error.response?.status
    });
    return Promise.reject(error);
  }
);

const API_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:5251';

if (!process.env.REACT_APP_API_URL) {
    console.warn('REACT_APP_API_URL is not set, using default');
}

const ConsistencyCheck: React.FC = () => {
  const [files, setFiles] = useState({
    plankart: null as File | null,
    bestemmelser: null as File | null,
    sosi: null as File | null
  });
  const [result, setResult] = useState<ConsistencyResult>({
    matching_fields: [],
    only_in_plankart: [],
    only_in_bestemmelser: [],
    only_in_sosi: [],
    is_consistent: false,
    document_fields: {
      plankart: {
        raw_fields: [],
        normalized_fields: [],
        text_sections: []
      },
      bestemmelser: {
        raw_fields: [],
        normalized_fields: [],
        text_sections: []
      }
    }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState<string>('');

  const handleFileChange = (type: keyof typeof files) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      const validationError = validateFile(file);
      
      if (validationError) {
        setError(validationError);
        return;
      }
      
      setFiles(prev => ({ ...prev, [type]: file }));
      setError(null); // Clear any previous errors
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      if (!files.plankart || !files.bestemmelser) {
        setError('Both Plankart and Bestemmelser files are required');
        return;
      }

      const formData = new FormData();
      formData.append('plankart', files.plankart);
      formData.append('bestemmelser', files.bestemmelser);
      if (files.sosi) {
        formData.append('sosi', files.sosi);
      }

      const response = await axios.post(
        `${API_URL}/api/check-field-consistency`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          }
        }
      );

      // The backend wraps the result in a "result" property
      const resultData = response.data.result;
      
      setResult({
        matching_fields: resultData.matching_fields || [],
        only_in_plankart: resultData.only_in_plankart || [],
        only_in_bestemmelser: resultData.only_in_bestemmelser || [],
        only_in_sosi: resultData.only_in_sosi || [],
        is_consistent: resultData.is_consistent || false,
        document_fields: {
          plankart: resultData.document_fields?.plankart || {
            raw_fields: [],
            normalized_fields: [],
            text_sections: []
          },
          bestemmelser: resultData.document_fields?.bestemmelser || {
            raw_fields: [],
            normalized_fields: [],
            text_sections: []
          },
          ...(resultData.document_fields?.sosi && {
            sosi: resultData.document_fields.sosi
          })
        }
      });

    } catch (err) {
      logger.error('Error during consistency check:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-white shadow-lg rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Feltsjekk for Reguleringsplan</h2>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(files).map(([type, file]) => (
              <div key={type} className="flex flex-col">
                <label className="block text-sm font-medium text-gray-700 mb-2 capitalize">
                  {type === 'plankart' ? 'Plankart' : 
                   type === 'bestemmelser' ? 'Bestemmelser' : 'SOSI-fil'} 
                  {type !== 'sosi' && <span className="text-red-500">*</span>}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    onChange={handleFileChange(type as keyof typeof files)}
                    accept=".pdf,.xml,.sos"
                    className="hidden"
                    id={`file-${type}`}
                  />
                  <label
                    htmlFor={`file-${type}`}
                    className="cursor-pointer flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                  >
                    {file ? (
                      <span className="truncate max-w-xs">{file.name}</span>
                    ) : (
                      <span>Velg fil</span>
                    )}
                  </label>
                </div>
              </div>
            ))}
          </div>

          {loading && (
            <div className="mt-4">
              <CircularProgress progress={progress} />
              <p className="text-sm text-gray-600 mt-2">{processingStep}</p>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded">
              <p className="text-red-700">{error}</p>
            </div>
          )}

          <div className="flex items-center justify-between">
            <button
              type="submit"
              disabled={loading || !files.plankart || !files.bestemmelser}
              className={`flex items-center justify-center px-6 py-3 border border-transparent rounded-md shadow-sm text-base font-medium text-white transition-colors
                ${loading || !files.plankart || !files.bestemmelser 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700'}`}
            >
              {loading ? (
                <>
                  <CircularProgress progress={progress} size={24} className="mr-2" />
                  <span>{processingStep}</span>
                </>
              ) : (
                'Start analyse'
              )}
            </button>
          </div>
        </form>
      </div>

      {result && !error && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">Analysis Results</h2>
          
          {/* Consistency Status */}
          <div className={`p-4 mb-4 rounded ${result.is_consistent ? 'bg-green-100' : 'bg-red-100'}`}>
            <h3 className="font-medium">
              {result.is_consistent ? 'No inconsistencies found' : 'Inconsistencies found'}
            </h3>
          </div>

          {/* Matching Fields */}
          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-500">Matching Fields</h4>
            <div className="mt-2 flex flex-wrap gap-2">
              {result.matching_fields.map(field => (
                <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  {field}
                </span>
              ))}
            </div>
          </div>

          {/* Fields only in Plankart */}
          {result.only_in_plankart.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-500">Only in Plankart</h4>
              <div className="mt-2 flex flex-wrap gap-2">
                {result.only_in_plankart.map(field => (
                  <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                    {field}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Fields only in Bestemmelser */}
          {result.only_in_bestemmelser.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-500">Only in Bestemmelser</h4>
              <div className="mt-2 flex flex-wrap gap-2">
                {result.only_in_bestemmelser.map(field => (
                  <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    {field}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          {result.metadata && Object.keys(result.metadata).length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-500">Metadata</h4>
              <div className="mt-2 grid grid-cols-2 gap-4">
                {Object.entries(result.metadata).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 p-2 rounded">
                    <span className="font-medium">{key.replace('_', ' ').toUpperCase()}: </span>
                    <span>{value || 'Not found'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fields Display */}
      <div className="mt-4 space-y-4">
        {result.document_fields && Object.entries(result.document_fields).map(([docType, fields]) => 
          fields && (
            <div key={docType} className="bg-white shadow rounded-lg p-4">
              <h3 className="text-lg font-medium text-gray-900 capitalize">{docType}</h3>
              
              {/* Raw Fields */}
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-500">Raw Fields</h4>
                <div className="mt-2 flex flex-wrap gap-2">
                  {fields.raw_fields.map(field => (
                    <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {field}
                    </span>
                  ))}
                </div>
              </div>

              {/* Normalized Fields */}
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-500">Normalized Fields</h4>
                <div className="mt-2 flex flex-wrap gap-2">
                  {fields.normalized_fields.map(field => (
                    <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      {field}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default ConsistencyCheck;