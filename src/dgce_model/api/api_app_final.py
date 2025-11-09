"""
Fixed API Application for DGCE UAE Tax Policy Simulator (FIXED)
==============================================================

FIXES:
- Proper error handling and validation
- Consistent integration with fixed modules
- Better parameter validation
- Realistic bounds checking
- Improved response formatting
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory

# Optional CORS support
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False

# Import sector mapping
try:
    from dgce_model.sector_mapping import ISIC_SECTORS, SHORT_TO_ISIC, get_policy_rate_for_sector
except ImportError:
    ISIC_SECTORS = [
        'Financial and insurance activities',
        'Manufacturing', 
        'Mining and quarrying',
        'Real estate activities',
        'Wholesale and retail trade, repair of motor vehicles and motorcycles',
        'Construction',
        'Information and communication',
        'Professional, scientific and technical activities',
        'Transportation and storage',
        'Accommodation and food service activities'
    ]
    SHORT_TO_ISIC = {
        'financial': 'Financial and insurance activities',
        'manufacturing': 'Manufacturing',
        'oil_gas': 'Mining and quarrying', 
        'real_estate': 'Real estate activities',
        'trade': 'Wholesale and retail trade, repair of motor vehicles and motorcycles',
        'construction': 'Construction',
        'technology': 'Information and communication',
        'professional': 'Professional, scientific and technical activities',
        'transport': 'Transportation and storage',
        'hospitality': 'Accommodation and food service activities'
    }
    
    def get_policy_rate_for_sector(sector, policy):
        return policy.get('standard_rate', 0.09)

# Import enhanced modules with proper error handling
try:
    from .quick_simulation_enhanced import EnhancedQuickSimulation
    QUICK_SIM_AVAILABLE = True
    print("✅ EnhancedQuickSimulation loaded")
except ImportError as e:
    print(f"⚠️ EnhancedQuickSimulation not available: {e}")
    QUICK_SIM_AVAILABLE = False

try:
    from .comprehensive_analysis_enhanced import ComprehensivePolicyAnalyzer
    COMPREHENSIVE_AVAILABLE = True
    print("✅ ComprehensivePolicyAnalyzer loaded")
except ImportError as e:
    print(f"⚠️ ComprehensivePolicyAnalyzer not available: {e}")
    COMPREHENSIVE_AVAILABLE = False

try:
    from .strategic_policy_scenarios import list_scenarios, run_scenario, get_scenario_params
    SCENARIOS_AVAILABLE = True
    print("✅ Strategic scenarios loaded")
except ImportError as e:
    print(f"⚠️ Strategic scenarios not available: {e}")
    SCENARIOS_AVAILABLE = False

try:
    from .sector_analyzer_enhanced import SectorPolicyAnalyzer
    SECTOR_ANALYZER_AVAILABLE = True
    print("✅ SectorPolicyAnalyzer loaded")
except ImportError as e:
    print(f"⚠️ SectorPolicyAnalyzer not available: {e}")
    SECTOR_ANALYZER_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


@app.route('/dashboard/<path:filename>')
def serve_dashboard(filename):
    """Serve static files from the dashboard directory."""
    return send_from_directory(DASHBOARD_DIR, filename)

# Enable CORS if available
if _CORS_AVAILABLE:
    CORS(app)

# Initialize analyzers
quick_sim = None
comprehensive_analyzer = None
sector_analyzer = None

try:
    if QUICK_SIM_AVAILABLE:
        quick_sim = EnhancedQuickSimulation()
        print("✅ Quick simulation initialized")
except Exception as e:
    print(f"⚠️ Failed to initialize quick simulation: {e}")

try:
    if COMPREHENSIVE_AVAILABLE:
        comprehensive_analyzer = ComprehensivePolicyAnalyzer()
        print("✅ Comprehensive analyzer initialized")
except Exception as e:
    print(f"⚠️ Failed to initialize comprehensive analyzer: {e}")

try:
    if SECTOR_ANALYZER_AVAILABLE:
        sector_analyzer = SectorPolicyAnalyzer()
        print("✅ Sector analyzer initialized")
except Exception as e:
    print(f"⚠️ Failed to initialize sector analyzer: {e}")


def safe_jsonify(data, status_code=200):
    """Safely convert data to JSON response."""
    try:
        # Convert numpy types to Python types
        def convert_numpy(obj):
            # Handle numpy scalar types
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        clean_data = convert_numpy(data)
        response = jsonify(clean_data)
        response.status_code = status_code
        return response
    except Exception as e:
        error_response = jsonify({'error': f'JSON serialization error: {str(e)}'})
        error_response.status_code = 500
        return error_response


def validate_simulation_params(data: Dict) -> Dict:
    """Validate simulation parameters with proper bounds."""
    errors = []
    
    # Required parameters
    required_params = ['standard_rate', 'compliance_rate']
    for param in required_params:
        if param not in data:
            errors.append(f"Missing required parameter: {param}")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    # Validate ranges
    try:
        standard_rate = float(data['standard_rate'])
        if not (0.0 <= standard_rate <= 0.5):
            errors.append("standard_rate must be between 0% and 50%")
    except (ValueError, TypeError):
        errors.append("standard_rate must be a valid number")
    
    try:
        compliance_rate = float(data['compliance_rate'])
        if not (0.1 <= compliance_rate <= 1.0):
            errors.append("compliance_rate must be between 10% and 100%")
    except (ValueError, TypeError):
        errors.append("compliance_rate must be a valid number")
    
    # Optional parameters with validation
    optional_params = {
        'threshold': (0, 1_000_000, 375_000),
        'small_biz_threshold': (1_000_000, 10_000_000, 3_000_000),
        'oil_gas_rate': (0.0, 0.8, 0.55),
        'fz_qualifying_rate': (0.0, 0.3, 0.0),
        'vat_rate': (0.0, 0.15, 0.05),
        'government_spending_rel_change': (-0.3, 0.3, 0.0),
        'years': (1, 20, 5)
    }
    
    for param, (min_val, max_val, default) in optional_params.items():
        if param in data:
            try:
                value = float(data[param])
                if not (min_val <= value <= max_val):
                    errors.append(f"{param} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{param} must be a valid number")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return data


@app.route('/', methods=['GET'])
def home():
    """API home endpoint."""
    return safe_jsonify({
        'message': 'DGCE UAE Tax Policy Simulator API (Fixed Version)',
        'version': '2.1.0-fixed',
        'status': 'operational',
        'endpoints': [
            '/api/simulate/quick',
            '/api/simulate/comprehensive', 
            '/health'
        ],
        'components': {
            'quick_simulation': QUICK_SIM_AVAILABLE,
            'comprehensive_analysis': COMPREHENSIVE_AVAILABLE,
            'sector_analysis': SECTOR_ANALYZER_AVAILABLE,
            'strategic_scenarios': SCENARIOS_AVAILABLE
        }
    })


@app.route('/api/simulate/quick', methods=['POST'])
def simulate_quick():
    """Quick policy simulation endpoint (FIXED)."""
    try:
        if not QUICK_SIM_AVAILABLE or quick_sim is None:
            return safe_jsonify({
                'error': 'Quick simulation not available',
                'message': 'The quick simulation module could not be loaded'
            }, 503)
        
        data = request.json or {}
        
        # Validate parameters
        validated_data = validate_simulation_params(data)
        
        print(f"🚀 Running quick simulation with rate {validated_data['standard_rate']*100:.1f}%")
        
        # Run simulation
        result = quick_sim.run_simulation(validated_data)
        
        # Add metadata
        result.update({
            'simulation_type': 'quick',
            'api_version': '2.1.0-fixed',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Quick simulation completed successfully")
        return safe_jsonify(result)
        
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        return safe_jsonify({
            'error': 'Validation error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }, 400)
        
    except Exception as e:
        print(f"❌ Quick simulation error: {str(e)}")
        return safe_jsonify({
            'error': 'Simulation failed',
            'message': str(e),
            'trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }, 500)


@app.route('/api/simulate/comprehensive', methods=['POST'])
def simulate_comprehensive():
    """Comprehensive policy analysis endpoint (FIXED)."""
    try:
        if not COMPREHENSIVE_AVAILABLE or comprehensive_analyzer is None:
            return safe_jsonify({
                'error': 'Comprehensive analysis not available',
                'message': 'The comprehensive analysis module could not be loaded'
            }, 503)
        
        data = request.json or {}
        
        # Validate parameters
        validated_data = validate_simulation_params(data)
        
        # Add time horizon if provided
        if 'time_horizon' in data:
            try:
                validated_data['time_horizon'] = max(1, min(20, int(data['time_horizon'])))
            except (ValueError, TypeError):
                validated_data['time_horizon'] = 10
        
        print(f"🔍 Running comprehensive analysis with rate {validated_data['standard_rate']*100:.1f}%")
        
        # Run comprehensive analysis
        result = comprehensive_analyzer.analyze_comprehensive_policy(validated_data)
        
        # Add metadata
        result.update({
            'simulation_type': 'comprehensive',
            'api_version': '2.1.0-fixed',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Comprehensive analysis completed successfully")
        return safe_jsonify(result)
        
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        return safe_jsonify({
            'error': 'Validation error', 
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }, 400)
        
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {str(e)}")
        return safe_jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }, 500)


@app.route('/api/simulate/sector', methods=['POST'])
def simulate_sector():
    """Sector-specific analysis endpoint (FIXED)."""
    try:
        if not SECTOR_ANALYZER_AVAILABLE or sector_analyzer is None:
            return safe_jsonify({
                'error': 'Sector analysis not available',
                'message': 'The sector analysis module could not be loaded'
            }, 503)
        
        data = request.json or {}
        
        # Validate sector parameter
        sector = data.get('sector')
        if not sector:
            return safe_jsonify({
                'error': 'Missing sector parameter',
                'available_sectors': ISIC_SECTORS
            }, 400)
        
        # Validate policy parameters
        policy_params = data.get('policy_params', {})
        if not policy_params:
            return safe_jsonify({
                'error': 'Missing policy_params'
            }, 400)
        
        validated_policy = validate_simulation_params(policy_params)
        
        print(f"🎯 Running sector analysis for {sector}")
        
        # Run sector analysis
        result = sector_analyzer.analyze_sector_policy(sector, validated_policy)
        
        # Add metadata
        result.update({
            'simulation_type': 'sector',
            'sector_analyzed': sector,
            'api_version': '2.1.0-fixed',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Sector analysis completed successfully")
        return safe_jsonify(result)
        
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        return safe_jsonify({
            'error': 'Validation error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }, 400)
        
    except Exception as e:
        print(f"❌ Sector analysis error: {str(e)}")
        return safe_jsonify({
            'error': 'Sector analysis failed',
            'message': str(e),
            'trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }, 500)


@app.route('/api/scenarios/list', methods=['GET'])
def list_policy_scenarios():
    """List available policy scenarios (FIXED)."""
    try:
        if not SCENARIOS_AVAILABLE:
            return safe_jsonify({
                'scenarios': [],
                'total': 0,
                'message': 'Strategic scenarios not available'
            })
        
        scenarios = list_scenarios()
        return safe_jsonify({
            'scenarios': scenarios,
            'total': len(scenarios),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error listing scenarios: {str(e)}")
        return safe_jsonify({
            'scenarios': [],
            'total': 0,
            'error': str(e)
        })


@app.route('/api/scenarios/run', methods=['POST'])
def run_policy_scenario():
    """Run a specific policy scenario (FIXED)."""
    try:
        if not SCENARIOS_AVAILABLE:
            return safe_jsonify({
                'error': 'Strategic scenarios not available'
            }, 503)
        
        data = request.json or {}
        scenario_name = data.get('scenario_name')
        
        if not scenario_name:
            return safe_jsonify({
                'error': 'Missing scenario_name parameter'
            }, 400)
        
        print(f"🎭 Running scenario: {scenario_name}")
        
        # Run scenario
        result = run_scenario(scenario_name)
        
        # Add metadata
        result.update({
            'simulation_type': 'scenario',
            'scenario_name': scenario_name,
            'api_version': '2.1.0-fixed',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Scenario completed successfully")
        return safe_jsonify(result)
        
    except Exception as e:
        print(f"❌ Scenario error: {str(e)}")
        return safe_jsonify({
            'error': 'Scenario execution failed',
            'message': str(e),
            'trace': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }, 500)


@app.route('/api/parameters/bounds', methods=['GET'])
def get_parameter_bounds():
    """Get valid parameter bounds for validation (FIXED)."""
    try:
        bounds = {
            'standard_rate': {
                'min': 0.0,
                'max': 0.5,
                'default': 0.09,
                'description': 'Standard corporate tax rate (0-50%)'
            },
            'compliance_rate': {
                'min': 0.1,
                'max': 1.0,
                'default': 0.75,
                'description': 'Expected compliance rate (10-100%)'
            },
            'threshold': {
                'min': 0,
                'max': 1_000_000,
                'default': 375_000,
                'description': 'Tax-free threshold in AED'
            },
            'small_biz_threshold': {
                'min': 1_000_000,
                'max': 10_000_000,
                'default': 3_000_000,
                'description': 'Small business relief threshold in AED'
            },
            'oil_gas_rate': {
                'min': 0.0,
                'max': 0.8,
                'default': 0.55,
                'description': 'Oil & gas sector tax rate (0-80%)'
            },
            'fz_qualifying_rate': {
                'min': 0.0,
                'max': 0.3,
                'default': 0.0,
                'description': 'Free zone qualifying income rate (0-30%)'
            }
        }
        
        return safe_jsonify({
            'parameter_bounds': bounds,
            'sectors': ISIC_SECTORS,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return safe_jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (FIXED)."""
    try:
        # Test basic functionality
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0-fixed',
            'components': {
                'api': 'operational',
                'quick_simulation': 'available' if QUICK_SIM_AVAILABLE and quick_sim else 'unavailable',
                'comprehensive_analysis': 'available' if COMPREHENSIVE_AVAILABLE and comprehensive_analyzer else 'unavailable',
                'sector_analysis': 'available' if SECTOR_ANALYZER_AVAILABLE and sector_analyzer else 'unavailable',
                'strategic_scenarios': 'available' if SCENARIOS_AVAILABLE else 'unavailable'
            },
            'data': {
                'sectors': len(ISIC_SECTORS),
                'scenarios': len(list_scenarios()) if SCENARIOS_AVAILABLE else 0
            }
        }
        
        # Determine overall health
        unavailable_components = [k for k, v in health_status['components'].items() if v == 'unavailable' and k != 'api']
        if len(unavailable_components) > len(health_status['components']) / 2:
            health_status['status'] = 'degraded'
        
        return safe_jsonify(health_status)
        
    except Exception as e:
        return safe_jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return safe_jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }, 404)


@app.errorhandler(500)
def internal_error(error):
    return safe_jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }, 500)


@app.errorhandler(400)
def bad_request(error):
    return safe_jsonify({
        'error': 'Bad request',
        'message': 'Invalid request format or parameters',
        'timestamp': datetime.now().isoformat()
    }, 400)


if __name__ == '__main__':
    print("🚀 Starting DGCE API Server (Fixed Version)...")
    print(f"📊 Available sectors: {len(ISIC_SECTORS)}")
    
    if SCENARIOS_AVAILABLE:
        try:
            scenario_count = len(list_scenarios())
            print(f"🎯 Strategic scenarios: {scenario_count}")
        except:
            print("🎯 Strategic scenarios: Error loading")
    
    print(f"🔧 Components status:")
    print(f"   Quick Simulation: {'✅' if QUICK_SIM_AVAILABLE else '❌'}")
    print(f"   Comprehensive Analysis: {'✅' if COMPREHENSIVE_AVAILABLE else '❌'}")
    print(f"   Sector Analysis: {'✅' if SECTOR_ANALYZER_AVAILABLE else '❌'}")
    print(f"   Strategic Scenarios: {'✅' if SCENARIOS_AVAILABLE else '❌'}")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
