# backend/api_helper.py
import os
import json
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def analyze_with_openai(feedback_data: Dict) -> Dict:
    """
    Send analysis data directly to OpenAI for AI coaching feedback
    
    Args:
        feedback_data: The feedback dict from AnalysisEngine with frame_data
        
    Returns:
        Enhanced feedback with AI insights
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found in environment variables")
        feedback_data['ai_enabled'] = False
        return feedback_data
    
    try:
        client = OpenAI(api_key=api_key)
        prompt = build_analysis_prompt(feedback_data)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert volleyball coach specializing in biomechanical analysis of serve technique. Provide specific, actionable feedback based on the data provided."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        ai_feedback = response.choices[0].message.content
        feedback_data['ai_analysis'] = ai_feedback
        feedback_data['ai_enabled'] = True
        
        return feedback_data
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        feedback_data['ai_enabled'] = False
        return feedback_data


def build_analysis_prompt(feedback_data: Dict) -> str:
    """Build detailed prompt with all biomechanical data"""
    
    frame_data = feedback_data.get('frame_data', {})
    phases = frame_data.get('phases', {})
    summary = frame_data.get('summary_stats', {})
    recommendations = feedback_data.get('recommendations', [])
    
    trophy_pose = phases.get('trophy_pose', {})
    ball_contact = phases.get('ball_contact', {})
    
    prompt = f"""Analyze this volleyball serve based on biomechanical data collected from video analysis:

**PHASE DETECTION:**

Trophy Pose (Frame {trophy_pose.get('frame', 'N/A')}):
- Elbow flexion: {trophy_pose.get('elbow_flexion', 'N/A')}°
- Wrist height: {trophy_pose.get('wrist_height', 'N/A')}
- Ideal elbow range: 90-110°

Ball Contact (Frame {ball_contact.get('frame', 'N/A')}):
- Shoulder abduction: {ball_contact.get('shoulder_abduction', 'N/A')}°
- Elbow extension: {ball_contact.get('elbow_extension', 'N/A')}°
- Max wrist velocity: {ball_contact.get('max_velocity', 'N/A')}
- Ideal shoulder: 120-150°, Ideal elbow: 170-180°

**KEY METRICS:**
- Minimum elbow angle during motion: {summary.get('min_elbow_angle', 'N/A')}° at frame {summary.get('min_elbow_frame', 'N/A')}
- Maximum wrist velocity: {summary.get('max_wrist_velocity', 'N/A')}
- Contact occurred at frame: {summary.get('contact_frame', 'N/A')}
- Total frames analyzed: {frame_data.get('total_frames', 'N/A')}

**AUTOMATED FEEDBACK DETECTED:**
{json.dumps(recommendations, indent=2)}

Based on this biomechanical analysis, provide:

1. **Overall Technique Rating** (1-10 scale with brief explanation)
2. **Top 3 Specific Improvements** (actionable coaching cues)
3. **Kinetic Chain Analysis** (comment on timing and sequencing from legs→hips→torso→arm)
4. **2-3 Recommended Drills** (specific exercises to address the weaknesses)

Format your response clearly with these sections. Be concise but specific. Focus on practical improvements the athlete can implement immediately."""
    
    return prompt