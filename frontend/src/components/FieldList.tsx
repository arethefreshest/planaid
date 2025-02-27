import React from 'react';

interface FieldListProps {
  title: string;
  fields: string[] | undefined;
}

const FieldList: React.FC<FieldListProps> = ({ title, fields }) => {
  if (!fields || fields.length === 0) {
    return null;
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>{title}</h3>
      <ul style={styles.list}>
        {fields.map((field, index) => (
          <li key={index} style={styles.listItem}>
            {field}
          </li>
        ))}
      </ul>
    </div>
  );
};

const styles = {
  container: {
    marginBottom: '20px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '10px',
  },
  list: {
    listStyleType: 'disc',
    paddingLeft: '20px',
  },
  listItem: {
    fontSize: '14px',
    color: '#555',
  },
};

export default FieldList;
