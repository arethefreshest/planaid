/**
 * Document Fields Display Component
 * 
 * Displays extracted fields from regulatory documents, showing both raw
 * and normalized versions of the fields.
 * 
 * Features:
 * - Separate display of raw and normalized fields
 * - Visual differentiation between field types
 * - Responsive layout
 */

import React from 'react';
import type { DocumentFields } from '../types';

interface Props {
  /** Type of document (plankart/bestemmelser/sosi) */
  documentType: string;
  /** Extracted fields from the document */
  fields: DocumentFields;
}

export const DocumentFieldsDisplay: React.FC<Props> = ({ documentType, fields }) => {
  return (
    <div className="mt-4 bg-white shadow rounded-lg p-4">
      <h3 className="text-lg font-medium text-gray-900 capitalize">{documentType}</h3>
      
      {/* Raw Fields Section */}
      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-500">Raw Fields</h4>
        <div className="mt-2 flex flex-wrap gap-2">
          {fields.raw_fields.map((field: string) => (
            <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              {field}
            </span>
          ))}
        </div>
      </div>

      {/* Normalized Fields Section */}
      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-500">Normalized Fields</h4>
        <div className="mt-2 flex flex-wrap gap-2">
          {fields.normalized_fields.map((field: string) => (
            <span key={field} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {field}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}; 