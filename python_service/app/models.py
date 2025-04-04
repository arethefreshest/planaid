"""
Data Models Module

This module defines the Pydantic models used for data validation and serialization
in the regulatory document processing system. These models represent the structure
of extracted fields and consistency check results.

Key Models:
- DocumentFields: Represents extracted fields from a single document
- ConsistencyResult: Represents the comparison results between multiple documents
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Set # Set is not accessed

class DocumentFields(BaseModel):
    """
    Represents the fields extracted from a single document.
    
    Contains both raw fields as extracted and their normalized versions
    for consistent comparison.
    
    Attributes:
        raw_fields (List[str]): Fields as originally extracted from the document
        normalized_fields (List[str]): Fields after normalization processing
    """
    raw_fields: List[str]
    normalized_fields: List[str]

class ConsistencyResult(BaseModel):
    """
    Represents the results of a consistency check between documents.
    
    Contains matching fields, differences between documents, and overall
    consistency status.
    
    Attributes:
        matching_fields (List[str]): Fields found in all documents
        only_in_plankart (List[str]): Fields unique to plankart
        only_in_bestemmelser (List[str]): Fields unique to bestemmelser
        only_in_sosi (List[str]): Fields unique to SOSI file
        is_consistent (bool): True if all documents contain the same fields
        document_fields (Dict[str, DocumentFields]): Raw and normalized fields per document
        metadata (Optional[Dict[str, str]]): Additional document metadata
    """
    matching_fields: List[str]
    only_in_plankart: List[str]
    only_in_bestemmelser: List[str]
    only_in_sosi: List[str]
    is_consistent: bool
    document_fields: Dict[str, DocumentFields]
    metadata: Optional[Dict[str, str]] = None

class NerResponse(BaseModel):
    """
    Represents the response from the NER service.
    
    Attributes:
        fields (List[str]): List of extracted fields from bestemmelser
    """
    fields: List[str] 