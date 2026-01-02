import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import pandas as pd
import requests
from collections import defaultdict
import numpy as np

# RAG imports
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import faiss

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Systems Biology Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RAG Configuration
Entrez.email = "kkokay07@gmail.com"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if 'rag_documents' not in st.session_state:
    st.session_state.rag_documents = []
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = faiss.IndexFlatL2(384)
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_genes' not in st.session_state:
    st.session_state.selected_genes = []
if 'gene_pathway_data' not in st.session_state:
    st.session_state.gene_pathway_data = None

# =====================================================
# HELPER FUNCTIONS
# =====================================================


@st.cache_data
def load_organisms():
    """Load available organisms from KEGG"""
    org_url = "https://rest.kegg.jp/list/organism"
    org_response = requests.get(org_url).text.strip().split('\n')

    org_dict = {}
    for line in org_response:
        parts = line.split('\t')
        if len(parts) >= 3:
            org_dict[parts[1]] = parts[2]

    return org_dict


@st.cache_data
def load_pathways(org_code):
    """Load pathways for a specific organism"""
    url = f"https://rest.kegg.jp/list/pathway/{org_code}"
    response = requests.get(url).text.strip()

    if not response:
        return {}

    pathway_dict = {}
    for line in response.split('\n'):
        if '\t' in line:
            pid, pname = line.split('\t')
            pathway_dict[pid.replace('path:', '')] = pname

    return pathway_dict


@st.cache_data
def load_all_genes(org_code):
    """Load all genes for organism with pathway info"""
    link_url = f"https://rest.kegg.jp/link/pathway/{org_code}"
    link_response = requests.get(link_url).text.strip().split('\n')

    records = []
    for line in link_response:
        if '\t' in line:
            gene, pathway = line.split('\t')
            records.append((
                gene.replace(f"{org_code}:", ""),
                pathway.replace("path:", "")
            ))

    df = pd.DataFrame(records, columns=["Gene ID", "Pathway ID"])
    pathway_map = load_pathways(org_code)
    df["Pathway Name"] = df["Pathway ID"].map(pathway_map)

    return df


def fetch_pubmed(term, max_results=5):
    """Fetch PubMed abstracts for RAG"""
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
        record = Entrez.read(handle)
        pmids = record.get("IdList", [])

        docs = []
        for pmid in pmids:
            fetch = Entrez.efetch(db="pubmed", id=pmid,
                                  rettype="abstract", retmode="text")
            docs.append(fetch.read())

        return docs
    except:
        return []


def build_rag_context(query, docs, limit=3000):
    """Build RAG context from documents"""
    if not docs:
        return "No literature found."

    embeddings = embedding_model.encode(docs)
    st.session_state.vector_index.reset()
    st.session_state.vector_index.add(np.array(embeddings))

    st.session_state.rag_documents = docs

    query_emb = embedding_model.encode([query])
    _, idx = st.session_state.vector_index.search(
        query_emb, k=min(3, len(docs)))

    context = "\n\n".join(st.session_state.rag_documents[i] for i in idx[0])

    return context[:limit]


def analyze_kegg_pathway(org_code, target_pathway, max_other_paths, full_df=None):
    """Analyze KEGG pathway data"""
    if full_df is None:
        full_df = load_all_genes(org_code)

    genes_in_target = full_df[full_df["Pathway ID"]
                              == target_pathway]["Gene ID"].unique()
    gene_to_pathways = full_df.groupby(
        "Gene ID")["Pathway ID"].apply(set).to_dict()

    result_dict = defaultdict(list)

    pathway_map = load_pathways(org_code)

    for gene in genes_in_target:
        other_paths = gene_to_pathways.get(gene, set()) - {target_pathway}
        if len(other_paths) <= max_other_paths:
            result_dict[len(other_paths)].append({
                "Gene ID": gene,
                "Other Pathways": [
                    f"{pid} ({pathway_map.get(pid, 'Unknown')})"
                    for pid in sorted(other_paths)
                ]
            })

    return result_dict, genes_in_target.tolist(), full_df


def analyze_gene(gene_id, full_df):
    """Analyze a specific gene - get all pathways it's involved in"""
    gene_pathways = full_df[full_df["Gene ID"] == gene_id]

    pathway_list = []
    for _, row in gene_pathways.iterrows():
        pathway_list.append({
            "Pathway ID": row["Pathway ID"],
            "Pathway Name": row["Pathway Name"]
        })

    return pd.DataFrame(pathway_list)


def calculate_hub_genes(df, min_pathways=3):
    """Calculate hub genes (high connectivity)"""
    gene_pathway_count = df.groupby("Gene ID")["Pathway ID"].nunique()
    hub_genes = gene_pathway_count[gene_pathway_count >=
                                   min_pathways].sort_values(ascending=False)
    return hub_genes


def simulate_knockout_impact(gene_id, df):
    """Simulate impact of gene knockout"""
    affected_pathways = df[df["Gene ID"] == gene_id]["Pathway ID"].unique()

    impact_score = len(affected_pathways)

    pathway_details = []
    for pid in affected_pathways:
        genes_in_pathway = df[df["Pathway ID"] == pid]["Gene ID"].nunique()
        pathway_details.append({
            "Pathway": pid,
            "Total Genes": genes_in_pathway,
            "Impact": f"1/{genes_in_pathway} genes affected",
            "Severity": 1.0 / genes_in_pathway
        })

    return impact_score, pathway_details


def build_graph_from_kegg(selected_pathways, full_df, pleiotropy_threshold):
    """Build graph structure from KEGG data (replaces Neo4j)"""
    nodes = []
    edges = []
    gene_to_pathways = defaultdict(set)

    # Get genes for selected pathways
    for pathway_id in selected_pathways:
        pathway_genes = full_df[full_df["Pathway ID"] == pathway_id]

        # Add pathway node
        pathway_name = pathway_genes.iloc[0]["Pathway Name"] if not pathway_genes.empty else pathway_id
        nodes.append({
            "id": pathway_id,
            "label": "Pathway",
            "properties": {"id": pathway_id, "name": pathway_name}
        })

        # Add gene nodes and edges
        for gene_id in pathway_genes["Gene ID"].unique():
            nodes.append({
                "id": gene_id,
                "label": "Gene",
                "properties": {"id": gene_id}
            })

            edges.append({
                "source": pathway_id,
                "target": gene_id,
                "type": "HAS_GENE"
            })

            gene_to_pathways[gene_id].add(pathway_id)

    # Add pathway-pathway connections through shared genes
    if len(selected_pathways) > 1:
        for gene_id, pathways in gene_to_pathways.items():
            if len(pathways) >= 2:
                pathway_list = list(pathways)
                for i in range(len(pathway_list)):
                    for j in range(i + 1, len(pathway_list)):
                        edges.append({
                            "source": pathway_list[i],
                            "target": pathway_list[j],
                            "type": "SHARED_GENES"
                        })

    # Deduplicate nodes
    unique_nodes = {}
    for node in nodes:
        unique_nodes[node["id"]] = node

    # Deduplicate edges
    unique_edges = []
    seen_edges = set()
    for edge in edges:
        edge_key = (edge["source"], edge["target"], edge["type"])
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_edges.append(edge)

    # Find shared genes
    shared_genes = [gene for gene,
                    pathways in gene_to_pathways.items() if len(pathways) >= 2]

    return {
        "nodes": list(unique_nodes.values()),
        "edges": unique_edges,
        "shared_genes": shared_genes,
        "gene_to_pathways": dict(gene_to_pathways)
    }

# =====================================================
# UI LAYOUT
# =====================================================


st.title(" Systems Biology Platform - Standalone Edition")

# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.header(" Configuration")

# Organism selection
org_dict = load_organisms()
org_options = [f"{code} - {name}" for code, name in sorted(org_dict.items())]
selected_org_display = st.sidebar.selectbox(
    "Select Organism",
    options=org_options,
    index=org_options.index(
        "hsa - Homo sapiens (human)") if "hsa - Homo sapiens (human)" in org_options else 0
)
selected_org_code = selected_org_display.split(" - ")[0].strip()
selected_org_name = selected_org_display.split(" - ", 1)[1]

# Load all gene-pathway data for organism
with st.spinner(f"Loading gene data for {selected_org_name}..."):
    full_df = load_all_genes(selected_org_code)
    st.session_state.gene_pathway_data = full_df

# Analysis Mode Selection
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    options=["Pathway-Centric", "Gene-Centric"],
    index=0
)

if analysis_mode == "Pathway-Centric":
    # Pathway selection
    pathway_dict = load_pathways(selected_org_code)
    pathway_options = [f"{pid} - {pname}" for pid,
                       pname in sorted(pathway_dict.items())]

    pathway_mode = st.sidebar.radio(
        "Pathway Selection Mode",
        options=["Single Pathway", "Multiple Pathways (Comparison)"],
        index=0
    )

    if pathway_mode == "Single Pathway":
        selected_pathway_display = st.sidebar.selectbox(
            "Select Pathway",
            options=pathway_options,
            index=0 if pathway_options else None
        )

        if selected_pathway_display:
            selected_pathway_id = selected_pathway_display.split(
                " - ")[0].strip()
            selected_pathway_name = selected_pathway_display.split(" - ", 1)[1]
            selected_pathways = [selected_pathway_id]
        else:
            st.error("No pathways available for selected organism")
            st.stop()
    else:
        selected_pathway_displays = st.sidebar.multiselect(
            "Select Pathways (max 5)",
            options=pathway_options,
            max_selections=5
        )

        if not selected_pathway_displays:
            st.warning("Please select at least one pathway")
            st.stop()

        selected_pathways = [p.split(" - ")[0].strip()
                             for p in selected_pathway_displays]
        selected_pathway_id = selected_pathways[0]
        selected_pathway_name = "Multiple Pathways"

else:  # Gene-Centric mode
    # Gene selection
    all_genes = sorted(full_df["Gene ID"].unique())
    selected_gene_id = st.sidebar.selectbox(
        "Select Gene",
        options=all_genes,
        index=0
    )

# Analysis parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Parameters")

max_other_pathways = st.sidebar.slider(
    "Max Other Pathways (Gene Filter)",
    min_value=0,
    max_value=10,
    value=3,
    help="Show genes in target pathway with at most N other pathways"
)

pleiotropy_threshold = st.sidebar.slider(
    "Pleiotropy Threshold",
    min_value=2,
    max_value=15,
    value=5,
    help="Minimum number of pathways to consider a gene pleiotropic"
)

hub_gene_threshold = st.sidebar.slider(
    "Hub Gene Threshold",
    min_value=2,
    max_value=10,
    value=3,
    help="Minimum pathways for hub gene detection"
)

gene_rag_limit = st.sidebar.number_input(
    "Gene RAG Analysis Limit",
    min_value=1,
    max_value=20,
    value=3,
    help="Number of genes to analyze with RAG (performance)"
)

# Node sizing option
use_proportional_sizing = st.sidebar.checkbox(
    "Proportional Node Sizing",
    value=True,
    help="Size nodes based on pathway connectivity"
)

# Run Analysis Button
run_analysis = st.sidebar.button(
    " Run Analysis", type="primary", use_container_width=True)

# =====================================================
# MAIN CONTENT AREA - TABS
# =====================================================

if analysis_mode == "Pathway-Centric":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Pathway Analysis",
        " Knowledge Graph",
        " Hub Genes & Pleiotropy",
        " Knockout Simulation",
        " RAG Gene Insights"
    ])
else:
    tab1, tab2, tab3 = st.tabs([
        " Gene Analysis",
        " Gene Network",
        " Gene Literature"
    ])

# =====================================================
# PATHWAY-CENTRIC ANALYSIS
# =====================================================

if analysis_mode == "Pathway-Centric":

    # TAB 1: PATHWAY ANALYSIS
    with tab1:
        if pathway_mode == "Single Pathway":
            st.header(f"Pathway Analysis: {selected_pathway_name}")
        else:
            st.header("Multi-Pathway Comparison")
            st.info(f"Analyzing {len(selected_pathways)} pathways")

        if run_analysis:
            with st.spinner("Analyzing pathway data..."):
                if pathway_mode == "Single Pathway":
                    result_dict, genes_in_target, _ = analyze_kegg_pathway(
                        selected_org_code,
                        selected_pathway_id,
                        max_other_pathways,
                        full_df
                    )

                    st.session_state.analysis_results = {
                        'result_dict': result_dict,
                        'genes': genes_in_target,
                        'df': full_df
                    }
                else:
                    all_results = {}
                    combined_genes = set()

                    for pathway_id in selected_pathways:
                        try:
                            result_dict, genes, _ = analyze_kegg_pathway(
                                selected_org_code, pathway_id, max_other_pathways, full_df
                            )
                            all_results[pathway_id] = {
                                'result_dict': result_dict,
                                'genes': genes,
                                'df': full_df
                            }
                            combined_genes.update(genes)
                        except Exception as e:
                            st.error(
                                f"Error analyzing pathway {pathway_id}: {str(e)}")
                            continue

                    if not all_results:
                        st.error("Failed to analyze any pathways")
                        st.stop()

                    pathway_gene_sets = {
                        pid: set(all_results[pid]['genes']) for pid in all_results.keys()}

                    if len(pathway_gene_sets) > 1:
                        shared_genes = set.intersection(
                            *pathway_gene_sets.values())
                    else:
                        shared_genes = list(pathway_gene_sets.values())[
                            0] if pathway_gene_sets else set()

                    st.session_state.analysis_results = {
                        'multi_pathway': True,
                        'all_results': all_results,
                        'shared_genes': list(shared_genes),
                        'genes': list(combined_genes),
                        'df': full_df
                    }

        if st.session_state.analysis_results:
            if st.session_state.analysis_results.get('multi_pathway'):
                all_results = st.session_state.analysis_results['all_results']
                shared_genes = st.session_state.analysis_results['shared_genes']

                st.success(f" Analyzed {len(selected_pathways)} pathways")
                st.metric("Shared Genes Across All Pathways",
                          len(shared_genes))

                if shared_genes:
                    with st.expander(" Shared Genes (appear in ALL selected pathways)"):
                        st.write(", ".join(shared_genes[:50]))
                        if len(shared_genes) > 50:
                            st.info(f"... and {len(shared_genes)-50} more")

                pathway_dict_display = load_pathways(selected_org_code)

                for pathway_id in selected_pathways:
                    if pathway_id in all_results:
                        pathway_name = pathway_dict_display.get(
                            pathway_id, pathway_id)
                        result_dict = all_results[pathway_id]['result_dict']
                        genes = all_results[pathway_id]['genes']

                        with st.expander(f" {pathway_name} ({len(genes)} genes)"):
                            for n in range(max_other_pathways + 1):
                                genes_list = result_dict.get(n, [])
                                if genes_list:
                                    st.write(
                                        f"**Genes with {n} other pathways:** {len(genes_list)}")
                                    df_display = pd.DataFrame([
                                        {
                                            "Gene ID": g['Gene ID'],
                                            "Other Pathways": ", ".join(g['Other Pathways'][:2]) +
                                            (f" ... (+{len(g['Other Pathways'])-2})" if len(
                                                g['Other Pathways']) > 2 else "")
                                        }
                                        for g in genes_list[:20]
                                    ])
                                    st.dataframe(df_display, width='stretch')
            else:
                result_dict = st.session_state.analysis_results['result_dict']
                genes_in_target = st.session_state.analysis_results['genes']

                st.success(
                    f" Found {len(genes_in_target)} genes in pathway {selected_pathway_id}")

                for n in range(max_other_pathways + 1):
                    with st.expander(f" Genes in target pathway + {n} other pathway(s) ({len(result_dict.get(n, []))} genes)"):
                        genes = result_dict.get(n, [])

                        if genes:
                            df_display = pd.DataFrame([
                                {
                                    "Gene ID": g['Gene ID'],
                                    "Other Pathways": ", ".join(g['Other Pathways'][:3]) +
                                    (f" ... (+{len(g['Other Pathways'])-3} more)" if len(
                                        g['Other Pathways']) > 3 else "")
                                }
                                for g in genes
                            ])
                            st.dataframe(df_display, width='stretch')
                        else:
                            st.info("No genes found in this category")
        else:
            st.info(" Click 'Run Analysis' in the sidebar to begin")

    # TAB 2: KNOWLEDGE GRAPH
    with tab2:
        st.header(" Interactive Knowledge Graph")

        # Build graph from KEGG data
        graph_data = build_graph_from_kegg(
            selected_pathways, full_df, pleiotropy_threshold)
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        shared_genes_graph = set(graph_data.get("shared_genes", []))
        gene_to_pathways = graph_data.get("gene_to_pathways", {})

        # Calculate pathway counts for all genes
        gene_pathway_counts = full_df.groupby(
            "Gene ID")["Pathway ID"].nunique().to_dict()

        # Build NetworkX graph
        G = nx.Graph()

        for node in nodes:
            node_id = node["id"]
            label = node["label"]

            if label == "Pathway":
                color = "#ff7f0e"  # Orange
                size = 30
                title = f"Pathway: {node_id}"
            else:  # Gene
                pathway_count = gene_pathway_counts.get(node_id, 1)

                # Size based on pathway count if enabled
                if use_proportional_sizing:
                    # Scale: 10 (min) to 40 (max) based on pathway count
                    size = min(10 + (pathway_count * 3), 40)
                else:
                    size = 15

                # Color priority: Shared > Pleiotropic > Regular
                if node_id in shared_genes_graph:
                    color = "#9467bd"  # Purple
                    title = f"Shared Gene: {node_id}\n(in {len(gene_to_pathways.get(node_id, set()))} selected pathways)\n(total: {pathway_count} pathways)"
                elif pathway_count >= pleiotropy_threshold:
                    color = "#d62728"  # Red
                    size = size * 1.2 if use_proportional_sizing else 22
                    title = f"Pleiotropic Gene: {node_id}\n(in {pathway_count} pathways total)"
                else:
                    color = "#2ca02c"  # Green
                    title = f"Gene: {node_id}\n(in {pathway_count} pathway(s))"

            G.add_node(node_id, label=node_id, color=color,
                       size=int(size), title=title)

        for edge in edges:
            if edge.get("type") == "SHARED_GENES":
                G.add_edge(edge["source"], edge["target"],
                           color='rgba(255,127,14,0.5)',
                           width=2,
                           dashes=[5, 5],
                           title="Connected through shared genes")
            else:
                G.add_edge(edge["source"], edge["target"])

        # Create PyVis network
        net = Network(height="700px", width="100%",
                      bgcolor="#ffffff", font_color="black")
        net.from_nx(G)

        if pathway_mode == "Multiple Pathways (Comparison)":
            net.repulsion(node_distance=250, central_gravity=0.15,
                          spring_length=250, spring_strength=0.04, damping=0.09)
        else:
            net.repulsion(node_distance=180, central_gravity=0.2,
                          spring_length=200, spring_strength=0.05, damping=0.09)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_path = tmp_file.name

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=750, scrolling=True)
        os.unlink(html_path)

        # Legend
        st.markdown("###  Legend")

        if pathway_mode == "Multiple Pathways (Comparison)":
            cols = st.columns(5)
            with cols[0]:
                st.markdown(" **Pathway**")
            with cols[1]:
                st.markdown(" **Gene (specific)**")
            with cols[2]:
                st.markdown(" **Shared Gene**")
            with cols[3]:
                st.markdown(f" **Pleiotropic (≥{pleiotropy_threshold})**")
            with cols[4]:
                st.markdown(" **Pathway Link**")

            st.info(
                f" **Node Size**: {'Proportional to pathway count' if use_proportional_sizing else 'Fixed size'}")
        else:
            cols = st.columns(3)
            with cols[0]:
                st.markdown(" **Pathway**")
            with cols[1]:
                st.markdown(" **Gene (specific)**")
            with cols[2]:
                st.markdown(f" **Pleiotropic (≥{pleiotropy_threshold})**")

            st.info(
                f" **Node Size**: {'Proportional to pathway count' if use_proportional_sizing else 'Fixed size'}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Nodes", len(nodes))
        with col2:
            st.metric("Total Edges", len(edges))

    # TAB 3: HUB GENES
    with tab3:
        st.header(" Hub Genes & Pleiotropy Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hub Genes (High Connectivity)")
            hub_genes = calculate_hub_genes(full_df, hub_gene_threshold)

            if not hub_genes.empty:
                hub_df = pd.DataFrame({
                    'Gene ID': hub_genes.index,
                    'Pathway Count': hub_genes.values
                })
                st.dataframe(hub_df, width='stretch')
                st.bar_chart(hub_genes.head(20))
            else:
                st.info(
                    f"No hub genes found with >= {hub_gene_threshold} pathways")

        with col2:
            st.subheader("Pleiotropic Genes")
            gene_pathway_counts = full_df.groupby(
                "Gene ID")["Pathway ID"].nunique()
            pleiotropic_genes = gene_pathway_counts[gene_pathway_counts >= pleiotropy_threshold].sort_values(
                ascending=False)

            if not pleiotropic_genes.empty:
                pleio_df = pd.DataFrame({
                    'Gene ID': pleiotropic_genes.index,
                    'Pathway Count': pleiotropic_genes.values
                })
                st.dataframe(pleio_df, width='stretch')
                st.bar_chart(pleiotropic_genes.head(20))
            else:
                st.info(
                    f"No pleiotropic genes found with >= {pleiotropy_threshold} pathways")

        # ML-ready features
        st.subheader(" ML-Ready Features")

        gene_features = full_df.groupby("Gene ID").agg({
            "Pathway ID": 'nunique',
        }).reset_index()
        gene_features.columns = ['Gene ID', 'Pathway Count']

        st.download_button(
            label=" Download Gene Features (CSV)",
            data=gene_features.to_csv(index=False),
            file_name=f"{selected_org_code}_gene_features.csv",
            mime="text/csv"
        )

    # TAB 4: KNOCKOUT SIMULATION
    with tab4:
        st.header(" Gene Knockout Impact Simulation")

        if st.session_state.analysis_results:
            genes_in_target = st.session_state.analysis_results.get(
                'genes', [])

            if not genes_in_target:
                st.warning(
                    "No gene data available. Please run pathway analysis first.")
            else:
                selected_gene = st.selectbox(
                    "Select Gene for Knockout Simulation",
                    options=genes_in_target
                )

                if st.button(" Simulate Knockout"):
                    impact_score, pathway_details = simulate_knockout_impact(
                        selected_gene, full_df)

                    st.metric("Impact Score (Affected Pathways)", impact_score)

                    st.subheader(
                        f"Pathways Affected by {selected_gene} Knockout")
                    impact_df = pd.DataFrame(pathway_details)
                    st.dataframe(impact_df, width='stretch')

                    # Visualization
                    st.bar_chart(impact_df.set_index("Pathway")["Severity"])
        else:
            st.info(" Run pathway analysis first")

    # TAB 5: RAG
    with tab5:
        st.header(" RAG-Enhanced Gene Insights")

        if st.session_state.analysis_results:
            genes_in_target = st.session_state.analysis_results.get(
                'genes', [])

            if not genes_in_target:
                st.warning(
                    "No genes found. Please run pathway analysis first.")
            else:
                st.subheader(
                    f" Pathway Biological Context: {selected_pathway_name}")

                if st.button("Fetch Pathway Context"):
                    with st.spinner("Fetching PubMed literature..."):
                        pathway_docs = fetch_pubmed(
                            f"{selected_pathway_name} pathway")

                        if pathway_docs:
                            context = build_rag_context(
                                f"biological role of {selected_pathway_name} pathway",
                                pathway_docs
                            )
                            st.text_area("Pathway Context",
                                         context, height=300)
                        else:
                            st.warning("No literature found for this pathway")

                st.markdown("---")

                st.subheader(
                    f" Gene-Level Analysis (Limited to {gene_rag_limit} genes)")
                st.info(
                    f"Gene-level analysis is limited to {gene_rag_limit} genes for performance.")

                analyzed_genes = genes_in_target[:gene_rag_limit]

                for gene in analyzed_genes:
                    with st.expander(f"Gene: {gene}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button(f"Fetch Biological Context", key=f"bio_{gene}"):
                                with st.spinner(f"Fetching context for {gene}..."):
                                    gene_docs = fetch_pubmed(
                                        f"{gene} {selected_org_name}")

                                    if gene_docs:
                                        context = build_rag_context(
                                            f"biological function of gene {gene}",
                                            gene_docs,
                                            limit=2000
                                        )
                                        st.text_area(
                                            "Biological Context", context, height=200, key=f"bio_text_{gene}")
                                    else:
                                        st.warning("No literature found")

                        with col2:
                            if st.button(f"Fetch Knockout Phenotype", key=f"ko_{gene}"):
                                with st.spinner(f"Fetching knockout data for {gene}..."):
                                    knockout_docs = fetch_pubmed(
                                        f"{gene} knockout {selected_org_name}")

                                    if knockout_docs:
                                        context = build_rag_context(
                                            f"phenotype of {gene} knockout",
                                            knockout_docs,
                                            limit=2500
                                        )
                                        st.text_area(
                                            "Knockout Phenotype", context, height=200, key=f"ko_text_{gene}")
                                    else:
                                        st.warning(
                                            "No knockout literature found")
        else:
            st.info(" Run pathway analysis first")

# =====================================================
# GENE-CENTRIC ANALYSIS
# =====================================================

else:  # Gene-Centric mode

    # TAB 1: GENE ANALYSIS
    with tab1:
        st.header(f" Gene Analysis: {selected_gene_id}")

        # Get pathways for this gene
        gene_pathways_df = analyze_gene(selected_gene_id, full_df)

        st.metric("Pathways Involved", len(gene_pathways_df))

        st.subheader("Pathway Memberships")
        st.dataframe(gene_pathways_df, width='stretch')

        # Pathway count classification
        pathway_count = len(gene_pathways_df)

        if pathway_count == 1:
            st.info(
                f" **Pathway-Specific Gene**: {selected_gene_id} is specific to one pathway")
        elif pathway_count < pleiotropy_threshold:
            st.warning(
                f"🟡 **Multi-Pathway Gene**: {selected_gene_id} participates in {pathway_count} pathways")
        else:
            st.error(
                f" **Pleiotropic Hub Gene**: {selected_gene_id} is involved in {pathway_count} pathways (≥ threshold)")

        # Download option
        st.download_button(
            label=" Download Gene Pathway Data (CSV)",
            data=gene_pathways_df.to_csv(index=False),
            file_name=f"{selected_gene_id}_pathways.csv",
            mime="text/csv"
        )

    # TAB 2: GENE NETWORK
    with tab2:
        st.header(f" Gene Network: {selected_gene_id}")

        # Get pathways for this gene
        gene_pathways = full_df[full_df["Gene ID"] ==
                                selected_gene_id]["Pathway ID"].unique()

        # Build graph
        graph_data = build_graph_from_kegg(
            list(gene_pathways), full_df, pleiotropy_threshold)
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        gene_to_pathways = graph_data.get("gene_to_pathways", {})

        # Calculate pathway counts
        gene_pathway_counts = full_df.groupby(
            "Gene ID")["Pathway ID"].nunique().to_dict()

        # Build NetworkX graph
        G = nx.Graph()

        for node in nodes:
            node_id = node["id"]
            label = node["label"]

            if label == "Pathway":
                color = "#ff7f0e"  # Orange
                size = 30
                title = f"Pathway: {node_id}"
            else:  # Gene
                pathway_count = gene_pathway_counts.get(node_id, 1)

                # Highlight selected gene
                if node_id == selected_gene_id:
                    color = "#e377c2"  # Pink - highlighted
                    size = 35
                    title = f"SELECTED GENE: {node_id}\n(in {pathway_count} pathways)"
                elif pathway_count >= pleiotropy_threshold:
                    color = "#d62728"  # Red
                    size = 22
                    title = f"Pleiotropic Gene: {node_id}\n(in {pathway_count} pathways)"
                else:
                    color = "#2ca02c"  # Green
                    size = 15
                    title = f"Gene: {node_id}\n(in {pathway_count} pathway(s))"

            G.add_node(node_id, label=node_id, color=color,
                       size=int(size), title=title)

        for edge in edges:
            G.add_edge(edge["source"], edge["target"])

        # Create PyVis network
        net = Network(height="700px", width="100%",
                      bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.repulsion(node_distance=200, central_gravity=0.2,
                      spring_length=220, spring_strength=0.05, damping=0.09)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_path = tmp_file.name

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=750, scrolling=True)
        os.unlink(html_path)

        # Legend
        st.markdown("###  Legend")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(" **Pathway**")
        with cols[1]:
            st.markdown(" **Selected Gene**")
        with cols[2]:
            st.markdown(" **Co-pathway Gene**")
        with cols[3]:
            st.markdown(f" **Pleiotropic (≥{pleiotropy_threshold})**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pathways", len(
                [n for n in nodes if n["label"] == "Pathway"]))
        with col2:
            st.metric("Co-pathway Genes",
                      len([n for n in nodes if n["label"] == "Gene"]) - 1)

    # TAB 3: GENE LITERATURE
    with tab3:
        st.header(f" Literature: {selected_gene_id}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Biological Function")
            if st.button("Fetch Biological Context"):
                with st.spinner(f"Fetching context for {selected_gene_id}..."):
                    gene_docs = fetch_pubmed(
                        f"{selected_gene_id} {selected_org_name}")

                    if gene_docs:
                        context = build_rag_context(
                            f"biological function of gene {selected_gene_id}",
                            gene_docs,
                            limit=3000
                        )
                        st.text_area("Biological Context", context, height=400)
                    else:
                        st.warning("No literature found")

        with col2:
            st.subheader("Knockout Phenotype")
            if st.button("Fetch Knockout Data"):
                with st.spinner(f"Fetching knockout data for {selected_gene_id}..."):
                    knockout_docs = fetch_pubmed(
                        f"{selected_gene_id} knockout {selected_org_name}")

                    if knockout_docs:
                        context = build_rag_context(
                            f"phenotype of {selected_gene_id} knockout",
                            knockout_docs,
                            limit=3000
                        )
                        st.text_area("Knockout Phenotype", context, height=400)
                    else:
                        st.warning("No knockout literature found")

# =====================================================
# FOOTER
# =====================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Systems Biology Platform v1.0**")

if st.sidebar.button(" Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# =====================================================
# FIXED FOOTER (ALWAYS VISIBLE)
# =====================================================
st.markdown("""
<style>
.main {
    padding-bottom: 80px;
}
#fixed-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(0,0,0,0.05);
    padding: 8px;
    font-size: 14px;
    z-index: 100;
}
</style>

<div id="fixed-footer">
    <p style="text-align:center;">
        <strong>Systems Biology Platform</strong> · Version 1.0 · © 2025 &nbsp;|&nbsp;
        <a href="https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en" target="_blank">
            Dr. Sandip Garai
        </a> ·
        <a href="https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao" target="_blank">
            Dr. Kanaka K K
        </a>        
        &nbsp;|&nbsp;
        <a href="mailto:drgaraislab@gmail.com">Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)
