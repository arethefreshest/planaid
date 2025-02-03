/** Represents extracted fields from a document */
export interface DocumentFields {
  raw_fields: string[];
  normalized_fields: string[];
  text_sections: string[];
}

/** Represents the consistency check result structure */
export interface ConsistencyResult {
  matching_fields: string[];
  only_in_plankart: string[];
  only_in_bestemmelser: string[];
  only_in_sosi: string[];
  is_consistent: boolean;
  document_fields: {
    plankart?: DocumentFields;
    bestemmelser?: DocumentFields;
    sosi?: DocumentFields;
  };
  metadata?: {
    plan_id?: string;
    plankart_dato?: string;
    bestemmelser_dato?: string;
    vedtatt_dato?: string;
  };
} 