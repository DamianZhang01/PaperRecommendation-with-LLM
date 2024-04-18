from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    authors = Column(String)
    abstract = Column(Text)
    year = Column(Integer)
    venue = Column(String)
    citations = Column(Text)  # Storing citations as a string for simplicity

# Setup SQLite Database
def setup_database(uri='sqlite:///papers.db'):
    engine = create_engine(uri)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        # Load the file line by line because each line is a separate JSON object
        for line in file:
            paper_data = json.loads(line)
            add_paper(paper_data)

def add_paper(session, paper_data):
    paper = Paper(
        id=paper_data.get('id'),
        title=paper_data.get('title'),
        authors=', '.join(paper_data.get('authors', [])),  # Join list of authors into a single string
        abstract=paper_data.get('abstract', ''),
        year=paper_data.get('year', None),
        venue=paper_data.get('venue', ''),
        citations=json.dumps(paper_data.get('citations', []))  # Convert list of citations to JSON string
    )
    session.add(paper)
    session.commit()

def get_paper_by_id(session, paper_id):
    paper = session.query(Paper).filter(Paper.id == paper_id).one_or_none()
    if paper:
        return {
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "year": paper.year,
            "venue": paper.venue,
            "citations": paper.citations
        }
    else:
        return None
    
def get_paper_by_title(session, paper_title):
    paper = session.query(Paper).filter(Paper.title == paper_title).one_or_none()
    if paper:
        return {
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "year": paper.year,
            "venue": paper.venue,
            "citations": paper.citations
        }
    else:
        return None

