import React from 'react';

/**
 * Circular Progress Component
 * 
 * A reusable circular progress indicator component that displays progress
 * as a circular ring with a percentage in the center.
 * 
 * Features:
 * - Customizable size and stroke width
 * - Animated progress updates
 * - Percentage display in center
 * - CSS class customization
 */

interface CircularProgressProps {
  /** Progress value from 0 to 100 */
  progress: number;
  /** Size of the circle in pixels */
  size?: number;
  /** Width of the progress stroke */
  strokeWidth?: number;
  /** Additional CSS classes */
  className?: string;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  progress,
  size = 44,
  strokeWidth = 4,
  className = ''
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className={`relative inline-flex ${className}`}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          className="text-gray-200"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className="text-blue-600 transition-all duration-300"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-sm font-medium">
        {Math.round(progress)}%
      </span>
    </div>
  );
}; 