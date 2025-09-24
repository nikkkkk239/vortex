"""
Dashboard and Reporting System
Interactive analytical dashboard generator and PDF medical report system
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO
import tempfile

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Report generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
except ImportError:
    logging.warning("ReportLab not available - PDF generation will be limited")

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from medical_analyzer import AnalysisResult, DiagnosticFinding, PatientContext

logger = logging.getLogger(__name__)


class MedicalDashboard:
    """
    Interactive medical visualization dashboard generator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Medical Dashboard
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.theme = config.get('dashboard_theme', 'medical')
        
        # Set up plotting style
        self._setup_plotting_style()
        
        logger.info("Initialized MedicalDashboard")
    
    def _setup_plotting_style(self):
        """Setup consistent plotting style"""
        # Set color palette for medical theme
        if self.theme == 'medical':
            self.color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
            self.background_color = '#F8F9FA'
            self.text_color = '#212529'
        else:
            self.color_palette = px.colors.qualitative.Set1
            self.background_color = 'white'
            self.text_color = 'black'
        
        # Configure matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.facecolor': self.background_color,
            'axes.facecolor': self.background_color,
            'text.color': self.text_color,
            'axes.labelcolor': self.text_color,
            'xtick.color': self.text_color,
            'ytick.color': self.text_color
        })
        
        # Configure seaborn
        sns.set_palette(self.color_palette)
    
    def create_dashboard(self, analysis_results: List[AnalysisResult],
                        patient_context: Optional[PatientContext] = None) -> Dict[str, Any]:
        """
        Create comprehensive analytical dashboard
        
        Args:
            analysis_results: List of analysis results
            patient_context: Optional patient context
            
        Returns:
            Dashboard data with visualizations
        """
        try:
            dashboard_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_analyses': len(analysis_results),
                    'patient_context': patient_context is not None
                },
                'summary_metrics': self._create_summary_metrics(analysis_results),
                'confidence_analysis': self._create_confidence_visualizations(analysis_results),
                'findings_analysis': self._create_findings_visualizations(analysis_results),
                'risk_assessment': self._create_risk_visualizations(analysis_results),
                'temporal_analysis': self._create_temporal_visualizations(analysis_results),
                'quantum_metrics': self._create_quantum_visualizations(analysis_results),
                'recommendations_summary': self._create_recommendations_summary(analysis_results)
            }
            
            logger.info(f"Created dashboard with {len(analysis_results)} analysis results")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {'error': str(e)}
    
    def _create_summary_metrics(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create summary metrics cards"""
        if not results:
            return {}
        
        # Calculate key metrics
        confidences = [r.confidence_scores.get('overall_confidence', 0.5) for r in results]
        avg_confidence = np.mean(confidences)
        
        quantum_enhanced = sum(1 for r in results if r.quantum_metrics.get('quantum_enhanced', False))
        
        high_risk_cases = sum(1 for r in results 
                             if r.risk_stratification.get('risk_category', '').lower() == 'high')
        
        processing_times = [r.processing_metadata.get('processing_time', 0) for r in results]
        avg_processing_time = np.mean(processing_times)
        
        # Count findings by type
        all_findings = []
        for result in results:
            all_findings.extend(result.primary_findings)
        
        findings_by_type = {}
        for finding in all_findings:
            finding_type = finding.description.split()[0].lower()  # Simple categorization
            findings_by_type[finding_type] = findings_by_type.get(finding_type, 0) + 1
        
        return {
            'total_analyses': len(results),
            'average_confidence': round(avg_confidence, 3),
            'quantum_enhanced_count': quantum_enhanced,
            'quantum_enhanced_percentage': round((quantum_enhanced / len(results)) * 100, 1),
            'high_risk_cases': high_risk_cases,
            'high_risk_percentage': round((high_risk_cases / len(results)) * 100, 1),
            'average_processing_time': round(avg_processing_time, 2),
            'total_findings': len(all_findings),
            'findings_by_type': findings_by_type,
            'confidence_distribution': {
                'high': sum(1 for c in confidences if c >= 0.8),
                'moderate': sum(1 for c in confidences if 0.6 <= c < 0.8),
                'low': sum(1 for c in confidences if c < 0.6)
            }
        }
    
    def _create_confidence_visualizations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create confidence analysis visualizations"""
        if not results:
            return {}
        
        # Extract confidence data
        confidence_data = []
        for i, result in enumerate(results):
            confidence_data.append({
                'analysis_id': i + 1,
                'overall_confidence': result.confidence_scores.get('overall_confidence', 0.5),
                'quantum_confidence': result.confidence_scores.get('quantum_confidence', 0.5),
                'llava_confidence': result.confidence_scores.get('llava_confidence', 0.5),
                'hybrid_confidence': result.confidence_scores.get('hybrid_confidence', 0.5),
                'timestamp': result.timestamp,
                'risk_category': result.risk_stratification.get('risk_category', 'Unknown')
            })
        
        df = pd.DataFrame(confidence_data)
        
        # Create confidence distribution histogram
        confidence_hist = self._create_confidence_histogram(df)
        
        # Create confidence comparison chart
        confidence_comparison = self._create_confidence_comparison(df)
        
        # Create confidence vs risk scatter plot
        confidence_risk_scatter = self._create_confidence_risk_scatter(df)
        
        return {
            'confidence_histogram': confidence_hist,
            'confidence_comparison': confidence_comparison,
            'confidence_risk_scatter': confidence_risk_scatter,
            'statistics': {
                'mean_confidence': df['overall_confidence'].mean(),
                'std_confidence': df['overall_confidence'].std(),
                'min_confidence': df['overall_confidence'].min(),
                'max_confidence': df['overall_confidence'].max()
            }
        }
    
    def _create_confidence_histogram(self, df: pd.DataFrame) -> str:
        """Create confidence distribution histogram"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['overall_confidence'],
            nbinsx=20,
            name='Overall Confidence',
            marker_color=self.color_palette[0],
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Confidence Score Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            showlegend=True,
            template='plotly_white',
            height=400
        )
        
        return fig.to_json()
    
    def _create_confidence_comparison(self, df: pd.DataFrame) -> str:
        """Create confidence comparison chart"""
        confidence_types = ['overall_confidence', 'quantum_confidence', 'llava_confidence', 'hybrid_confidence']
        confidence_means = [df[conf_type].mean() for conf_type in confidence_types]
        confidence_labels = ['Overall', 'Quantum', 'LLaVA', 'Hybrid']
        
        fig = go.Figure(data=[
            go.Bar(
                x=confidence_labels,
                y=confidence_means,
                marker_color=self.color_palette[:len(confidence_labels)],
                text=[f'{val:.2f}' for val in confidence_means],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Average Confidence Scores by Component',
            xaxis_title='Confidence Type',
            yaxis_title='Average Confidence',
            template='plotly_white',
            height=400
        )
        
        return fig.to_json()
    
    def _create_confidence_risk_scatter(self, df: pd.DataFrame) -> str:
        """Create confidence vs risk scatter plot"""
        risk_mapping = {'Low': 1, 'Moderate': 2, 'High': 3, 'Unknown': 0}
        df['risk_numeric'] = df['risk_category'].map(risk_mapping)
        
        fig = px.scatter(
            df,
            x='overall_confidence',
            y='risk_numeric',
            color='risk_category',
            title='Confidence vs Risk Assessment',
            labels={
                'overall_confidence': 'Overall Confidence',
                'risk_numeric': 'Risk Level',
                'risk_category': 'Risk Category'
            },
            height=400
        )
        
        fig.update_yaxis(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Unknown', 'Low', 'Moderate', 'High']
        )
        
        return fig.to_json()
    
    def _create_findings_visualizations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create findings analysis visualizations"""
        if not results:
            return {}
        
        # Collect all findings
        findings_data = []
        for result in results:
            for finding in result.primary_findings:
                findings_data.append({
                    'description': finding.description,
                    'confidence': finding.confidence,
                    'finding_type': self._categorize_finding(finding.description),
                    'session_id': result.session_id,
                    'timestamp': result.timestamp
                })
        
        if not findings_data:
            return {}
        
        df = pd.DataFrame(findings_data)
        
        # Create findings frequency chart
        findings_frequency = self._create_findings_frequency_chart(df)
        
        # Create findings confidence chart
        findings_confidence = self._create_findings_confidence_chart(df)
        
        # Create findings timeline
        findings_timeline = self._create_findings_timeline(df)
        
        return {
            'findings_frequency': findings_frequency,
            'findings_confidence': findings_confidence,
            'findings_timeline': findings_timeline,
            'statistics': {
                'total_findings': len(df),
                'unique_finding_types': df['finding_type'].nunique(),
                'most_common_finding': df['finding_type'].mode().iloc[0] if not df.empty else 'None'
            }
        }
    
    def _categorize_finding(self, description: str) -> str:
        """Categorize finding based on description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['normal', 'unremarkable', 'no acute']):
            return 'Normal'
        elif any(word in description_lower for word in ['pneumonia', 'consolidation', 'infiltrate']):
            return 'Pneumonia'
        elif any(word in description_lower for word in ['fracture', 'break', 'crack']):
            return 'Fracture'
        elif any(word in description_lower for word in ['mass', 'tumor', 'lesion', 'nodule']):
            return 'Mass/Lesion'
        elif any(word in description_lower for word in ['fluid', 'effusion', 'edema']):
            return 'Fluid/Edema'
        else:
            return 'Other'
    
    def _create_findings_frequency_chart(self, df: pd.DataFrame) -> str:
        """Create findings frequency chart"""
        frequency_counts = df['finding_type'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=frequency_counts.index,
                y=frequency_counts.values,
                marker_color=self.color_palette[:len(frequency_counts)],
                text=frequency_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Finding Types Frequency',
            xaxis_title='Finding Type',
            yaxis_title='Count',
            template='plotly_white',
            height=400
        )
        
        return fig.to_json()
    
    def _create_findings_confidence_chart(self, df: pd.DataFrame) -> str:
        """Create findings confidence by type chart"""
        fig = px.box(
            df,
            x='finding_type',
            y='confidence',
            title='Confidence Distribution by Finding Type',
            height=400
        )
        
        fig.update_layout(template='plotly_white')
        return fig.to_json()
    
    def _create_findings_timeline(self, df: pd.DataFrame) -> str:
        """Create findings timeline"""
        if df.empty:
            return "{}"
        
        # Group by date and finding type
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        timeline_data = df.groupby(['date', 'finding_type']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data,
            x='date',
            y='count',
            color='finding_type',
            title='Findings Timeline',
            height=400
        )
        
        fig.update_layout(template='plotly_white')
        return fig.to_json()
    
    def _create_risk_visualizations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create risk assessment visualizations"""
        if not results:
            return {}
        
        # Extract risk data
        risk_data = []
        for result in results:
            risk_data.append({
                'risk_category': result.risk_stratification.get('risk_category', 'Unknown'),
                'risk_score': result.risk_stratification.get('risk_score', 0.0),
                'confidence': result.confidence_scores.get('overall_confidence', 0.5),
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(risk_data)
        
        # Create risk distribution pie chart
        risk_distribution = self._create_risk_distribution_chart(df)
        
        # Create risk score histogram
        risk_score_hist = self._create_risk_score_histogram(df)
        
        return {
            'risk_distribution': risk_distribution,
            'risk_score_histogram': risk_score_hist,
            'statistics': {
                'high_risk_percentage': (df['risk_category'] == 'High').mean() * 100,
                'average_risk_score': df['risk_score'].mean()
            }
        }
    
    def _create_risk_distribution_chart(self, df: pd.DataFrame) -> str:
        """Create risk distribution pie chart"""
        risk_counts = df['risk_category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.3,
            marker_colors=self.color_palette[:len(risk_counts)]
        )])
        
        fig.update_layout(
            title='Risk Category Distribution',
            height=400,
            template='plotly_white'
        )
        
        return fig.to_json()
    
    def _create_risk_score_histogram(self, df: pd.DataFrame) -> str:
        """Create risk score histogram"""
        fig = go.Figure(data=[
            go.Histogram(
                x=df['risk_score'],
                nbinsx=20,
                marker_color=self.color_palette[0],
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title='Risk Score Distribution',
            xaxis_title='Risk Score',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig.to_json()
    
    def _create_temporal_visualizations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create temporal analysis visualizations"""
        if not results:
            return {}
        
        # Extract temporal data
        temporal_data = []
        for result in results:
            temporal_data.append({
                'timestamp': result.timestamp,
                'confidence': result.confidence_scores.get('overall_confidence', 0.5),
                'processing_time': result.processing_metadata.get('processing_time', 0),
                'quantum_enhanced': result.quantum_metrics.get('quantum_enhanced', False)
            })
        
        df = pd.DataFrame(temporal_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create confidence over time chart
        confidence_timeline = self._create_confidence_timeline(df)
        
        # Create processing time chart
        processing_time_chart = self._create_processing_time_chart(df)
        
        return {
            'confidence_timeline': confidence_timeline,
            'processing_time_chart': processing_time_chart
        }
    
    def _create_confidence_timeline(self, df: pd.DataFrame) -> str:
        """Create confidence timeline"""
        fig = px.line(
            df,
            x='timestamp',
            y='confidence',
            title='Confidence Scores Over Time',
            height=400
        )
        
        fig.update_layout(template='plotly_white')
        return fig.to_json()
    
    def _create_processing_time_chart(self, df: pd.DataFrame) -> str:
        """Create processing time chart"""
        fig = px.scatter(
            df,
            x='timestamp',
            y='processing_time',
            color='quantum_enhanced',
            title='Processing Time Over Time',
            height=400
        )
        
        fig.update_layout(template='plotly_white')
        return fig.to_json()
    
    def _create_quantum_visualizations(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create quantum metrics visualizations"""
        if not results:
            return {}
        
        # Extract quantum data
        quantum_data = []
        for result in results:
            quantum_metrics = result.quantum_metrics
            quantum_data.append({
                'quantum_enhanced': quantum_metrics.get('quantum_enhanced', False),
                'confidence': result.confidence_scores.get('quantum_confidence', 0.5),
                'uncertainty': quantum_metrics.get('uncertainty_metrics', {}).get('quantum_uncertainty', 0.5),
                'entropy': quantum_metrics.get('uncertainty_metrics', {}).get('entropy', 0.5)
            })
        
        df = pd.DataFrame(quantum_data)
        
        # Create quantum enhancement impact chart
        quantum_impact = self._create_quantum_impact_chart(df)
        
        # Create quantum uncertainty chart
        quantum_uncertainty = self._create_quantum_uncertainty_chart(df)
        
        return {
            'quantum_impact': quantum_impact,
            'quantum_uncertainty': quantum_uncertainty,
            'statistics': {
                'quantum_enhanced_percentage': df['quantum_enhanced'].mean() * 100,
                'average_quantum_confidence': df['confidence'].mean()
            }
        }
    
    def _create_quantum_impact_chart(self, df: pd.DataFrame) -> str:
        """Create quantum enhancement impact chart"""
        enhanced_conf = df[df['quantum_enhanced']]['confidence'].mean()
        regular_conf = df[~df['quantum_enhanced']]['confidence'].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Regular Processing', 'Quantum Enhanced'],
                y=[regular_conf, enhanced_conf],
                marker_color=self.color_palette[:2],
                text=[f'{regular_conf:.2f}', f'{enhanced_conf:.2f}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Quantum Enhancement Impact on Confidence',
            yaxis_title='Average Confidence',
            template='plotly_white',
            height=400
        )
        
        return fig.to_json()
    
    def _create_quantum_uncertainty_chart(self, df: pd.DataFrame) -> str:
        """Create quantum uncertainty visualization"""
        fig = px.scatter(
            df,
            x='uncertainty',
            y='confidence',
            color='quantum_enhanced',
            title='Quantum Uncertainty vs Confidence',
            height=400
        )
        
        fig.update_layout(template='plotly_white')
        return fig.to_json()
    
    def _create_recommendations_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Create recommendations summary"""
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        sorted_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_recommendations': len(all_recommendations),
            'unique_recommendations': len(recommendation_counts),
            'top_recommendations': sorted_recommendations[:10],
            'recommendation_frequency': recommendation_counts
        }
    
    def export_dashboard(self, dashboard_data: Dict[str, Any], 
                        output_path: str, format: str = 'html') -> str:
        """
        Export dashboard to file
        
        Args:
            dashboard_data: Dashboard data
            output_path: Output file path
            format: Export format ('html', 'json', 'pdf')
            
        Returns:
            Path to exported file
        """
        try:
            output_path = Path(output_path)
            
            if format.lower() == 'html':
                return self._export_to_html(dashboard_data, output_path)
            elif format.lower() == 'json':
                return self._export_to_json(dashboard_data, output_path)
            elif format.lower() == 'pdf':
                return self._export_to_pdf(dashboard_data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            return ""
    
    def _export_to_html(self, dashboard_data: Dict[str, Any], output_path: Path) -> str:
        """Export dashboard to HTML"""
        html_content = self._generate_html_dashboard(dashboard_data)
        
        with open(output_path.with_suffix('.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path.with_suffix('.html'))
    
    def _export_to_json(self, dashboard_data: Dict[str, Any], output_path: Path) -> str:
        """Export dashboard to JSON"""
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return str(output_path.with_suffix('.json'))
    
    def _export_to_pdf(self, dashboard_data: Dict[str, Any], output_path: Path) -> str:
        """Export dashboard to PDF"""
        # This would require more complex PDF generation
        # For now, return JSON export
        return self._export_to_json(dashboard_data, output_path.with_suffix('.pdf'))
    
    def _generate_html_dashboard(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML dashboard"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard-section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .metric-card { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 150px; }
                .chart-container { margin: 20px 0; }
                h1, h2 { color: #2E86AB; }
                .metric-value { font-size: 24px; font-weight: bold; color: #A23B72; }
                .metric-label { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <h1>Medical Analysis Dashboard</h1>
            
            <div class="dashboard-section">
                <h2>Summary Metrics</h2>
                <div id="summary-metrics"></div>
            </div>
            
            <div class="dashboard-section">
                <h2>Confidence Analysis</h2>
                <div id="confidence-charts"></div>
            </div>
            
            <div class="dashboard-section">
                <h2>Findings Analysis</h2>
                <div id="findings-charts"></div>
            </div>
            
            <div class="dashboard-section">
                <h2>Risk Assessment</h2>
                <div id="risk-charts"></div>
            </div>
            
            <div class="dashboard-section">
                <h2>Quantum Metrics</h2>
                <div id="quantum-charts"></div>
            </div>
            
            <script>
                // Dashboard data
                const dashboardData = """ + json.dumps(dashboard_data, default=str) + """;
                
                // Render summary metrics
                function renderSummaryMetrics() {
                    const container = document.getElementById('summary-metrics');
                    const metrics = dashboardData.summary_metrics;
                    
                    if (metrics) {
                        container.innerHTML = `
                            <div class="metric-card">
                                <div class="metric-value">${metrics.total_analyses || 0}</div>
                                <div class="metric-label">Total Analyses</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${(metrics.average_confidence * 100).toFixed(1)}%</div>
                                <div class="metric-label">Average Confidence</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${metrics.quantum_enhanced_percentage || 0}%</div>
                                <div class="metric-label">Quantum Enhanced</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${metrics.high_risk_percentage || 0}%</div>
                                <div class="metric-label">High Risk Cases</div>
                            </div>
                        `;
                    }
                }
                
                // Render charts
                function renderCharts() {
                    const sections = [
                        { id: 'confidence-charts', data: dashboardData.confidence_analysis },
                        { id: 'findings-charts', data: dashboardData.findings_analysis },
                        { id: 'risk-charts', data: dashboardData.risk_assessment },
                        { id: 'quantum-charts', data: dashboardData.quantum_metrics }
                    ];
                    
                    sections.forEach(section => {
                        const container = document.getElementById(section.id);
                        if (section.data && container) {
                            let chartIndex = 0;
                            Object.entries(section.data).forEach(([key, value]) => {
                                if (typeof value === 'string' && value.startsWith('{')) {
                                    try {
                                        const chartData = JSON.parse(value);
                                        const chartDiv = document.createElement('div');
                                        chartDiv.id = `${section.id}-${chartIndex}`;
                                        chartDiv.className = 'chart-container';
                                        container.appendChild(chartDiv);
                                        Plotly.newPlot(chartDiv.id, chartData.data, chartData.layout, {responsive: true});
                                        chartIndex++;
                                    } catch (e) {
                                        console.error('Error parsing chart data:', e);
                                    }
                                }
                            });
                        }
                    });
                }
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    renderSummaryMetrics();
                    renderCharts();
                });
            </script>
        </body>
        </html>
        """
        
        return html_template


class MedicalReportGenerator:
    """
    PDF Medical Report Generator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Medical Report Generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.clinic_info = config.get('clinic_info', {})
        
        logger.info("Initialized MedicalReportGenerator")
    
    def generate_medical_report(self, analysis_result: AnalysisResult,
                              patient_info: Dict[str, Any],
                              clinic_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive medical report
        
        Args:
            analysis_result: Analysis result data
            patient_info: Patient information
            clinic_info: Optional clinic information
            
        Returns:
            Path to generated PDF report
        """
        try:
            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                pdf_path = temp_file.name
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Add custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=colors.HexColor('#2E86AB')
            )
            
            # Header
            story.append(Paragraph("QUANTUM-ENHANCED MEDICAL IMAGING ANALYSIS REPORT", title_style))
            story.append(Spacer(1, 20))
            
            # Add clinic information
            clinic_data = clinic_info or self.clinic_info
            if clinic_data:
                clinic_table_data = [
                    ['Clinic:', clinic_data.get('name', 'Not specified')],
                    ['Address:', clinic_data.get('address', 'Not specified')],
                    ['Phone:', clinic_data.get('phone', 'Not specified')],
                    ['Email:', clinic_data.get('email', 'Not specified')]
                ]
                clinic_table = Table(clinic_table_data, colWidths=[1.5*inch, 4*inch])
                clinic_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
                ]))
                story.append(clinic_table)
                story.append(Spacer(1, 20))
            
            # Patient information
            story.append(Paragraph("PATIENT INFORMATION", styles['Heading2']))
            patient_table_data = [
                ['Patient Name:', patient_info.get('name', 'Not specified')],
                ['Patient ID:', patient_info.get('id', 'Not specified')],
                ['Date of Birth:', patient_info.get('dob', 'Not specified')],
                ['Gender:', patient_info.get('gender', 'Not specified')],
                ['Study Date:', analysis_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')]
            ]
            patient_table = Table(patient_table_data, colWidths=[1.5*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Analysis summary
            story.append(Paragraph("ANALYSIS SUMMARY", styles['Heading2']))
            
            summary_data = [
                ['Imaging Modality:', analysis_result.imaging_modality.value.upper()],
                ['Session ID:', analysis_result.session_id],
                ['Processing Time:', f"{analysis_result.processing_metadata.get('processing_time', 0):.2f} seconds"],
                ['Quantum Enhancement:', 'Enabled' if analysis_result.quantum_metrics.get('quantum_enhanced') else 'Disabled']
            ]
            
            summary_table = Table(summary_data, colWidths=[1.8*inch, 3.7*inch])
            summary_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Primary findings
            story.append(Paragraph("PRIMARY FINDINGS", styles['Heading2']))
            
            for i, finding in enumerate(analysis_result.primary_findings, 1):
                finding_text = f"{i}. {finding.description}"
                story.append(Paragraph(finding_text, styles['Normal']))
                
                if finding.confidence:
                    confidence_text = f"   Confidence: {finding.confidence:.2f} ({self._get_confidence_category(finding.confidence)})"
                    story.append(Paragraph(confidence_text, styles['Normal']))
                
                if finding.recommendations:
                    story.append(Paragraph("   Recommendations:", styles['Normal']))
                    for rec in finding.recommendations:
                        story.append(Paragraph(f"   • {rec}", styles['Normal']))
                
                story.append(Spacer(1, 10))
            
            # Confidence metrics
            story.append(Paragraph("CONFIDENCE METRICS", styles['Heading2']))
            
            confidence_data = [
                ['Overall Confidence:', f"{analysis_result.confidence_scores.get('overall_confidence', 0):.2f}"],
                ['Quantum Confidence:', f"{analysis_result.confidence_scores.get('quantum_confidence', 0):.2f}"],
                ['LLaVA Confidence:', f"{analysis_result.confidence_scores.get('llava_confidence', 0):.2f}"],
                ['Confidence Level:', analysis_result.confidence_scores.get('confidence_level', 'Unknown')]
            ]
            
            confidence_table = Table(confidence_data, colWidths=[2*inch, 3.5*inch])
            confidence_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(confidence_table)
            story.append(Spacer(1, 20))
            
            # Risk assessment
            story.append(Paragraph("RISK ASSESSMENT", styles['Heading2']))
            
            risk_data = analysis_result.risk_stratification
            risk_table_data = [
                ['Risk Category:', risk_data.get('risk_category', 'Unknown')],
                ['Risk Score:', f"{risk_data.get('risk_score', 0):.2f}"]
            ]
            
            if risk_data.get('risk_factors'):
                risk_factors_text = ', '.join(risk_data['risk_factors'])
                risk_table_data.append(['Risk Factors:', risk_factors_text])
            
            risk_table = Table(risk_table_data, colWidths=[1.5*inch, 4*inch])
            risk_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 20))
            
            # Clinical recommendations
            story.append(Paragraph("CLINICAL RECOMMENDATIONS", styles['Heading2']))
            
            for i, recommendation in enumerate(analysis_result.recommendations, 1):
                story.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
            
            story.append(Spacer(1, 30))
            
            # Footer
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.gray
            )
            
            story.append(Paragraph("IMPORTANT DISCLAIMER:", styles['Heading3']))
            story.append(Paragraph(
                "This AI-generated report is intended to assist healthcare professionals and should not replace clinical judgment. "
                "All findings should be correlated with patient history, physical examination, and other diagnostic information. "
                "The AI system provides probability-based assessments that require professional medical interpretation.",
                footer_style
            ))
            
            story.append(Spacer(1, 20))
            story.append(Paragraph(
                f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Quantum-Enhanced Medical Imaging AI System v1.0",
                footer_style
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Generated medical report: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating medical report: {str(e)}")
            return ""
    
    def _get_confidence_category(self, confidence: float) -> str:
        """Get confidence category text"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"