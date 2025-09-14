
export interface Query {
  id: number;
  term: string;
  field: string;
}

export interface FilterState {
  fromDate: string;
  toDate: string;
  publicationTypes: string[];
}

export interface PubMedArticle {
  uid: string;
  pubdate: string;
  title: string;
  authors: { name: string }[];
  source: string; // Journal name
  abstract?: string;
}
