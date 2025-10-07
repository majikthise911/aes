import openai
import os
from typing import List, Dict

class ProposalChatbot:
    """AI chatbot that can answer questions about uploaded proposals."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.conversation_history = []

    def prepare_context(self, proposals: List[Dict]) -> str:
        """Prepare context from proposals for the chatbot."""
        if not proposals:
            return "No proposals have been uploaded yet."

        context_parts = []

        for i, proposal in enumerate(proposals, 1):
            epc = proposal.get('epc_contractor', {})
            costs = proposal.get('costs', {})
            equipment = proposal.get('equipment', {})
            capacity = proposal.get('capacity', {})
            scope = proposal.get('scope', {})
            project_info = proposal.get('project_info', {})
            detailed_sov = proposal.get('detailed_sov', {})

            proposal_context = f"""
PROPOSAL #{i}: {epc.get('company_name', 'Unknown EPC')}
================================================
PROJECT: {project_info.get('project_name', 'N/A')}
LOCATION: {project_info.get('location', {}).get('state', 'N/A')}
TECHNOLOGY: {proposal.get('technology', {}).get('type', 'N/A')}
CAPACITY: {capacity.get('ac_mw', 'N/A')} MW AC, {capacity.get('dc_mw', 'N/A')} MW DC

COSTS:
- Total: ${costs.get('total_project_cost', 'N/A'):,.0f} if costs.get('total_project_cost') else 'N/A'
- Cost per Watt DC: ${costs.get('cost_per_watt_dc', 'N/A')}

EQUIPMENT:
- Modules: {equipment.get('modules', {}).get('manufacturer', 'N/A')} {equipment.get('modules', {}).get('model', '')} ({equipment.get('modules', {}).get('wattage', 'N/A')}W)
- Inverters: {equipment.get('inverters', {}).get('manufacturer', 'N/A')} {equipment.get('inverters', {}).get('model', '')}
- Racking: {equipment.get('racking', {}).get('manufacturer', 'N/A')} {equipment.get('racking', {}).get('type', '')}

SCOPE SUMMARY:
- Assumptions: {len(scope.get('assumptions', []))} items
- Exclusions: {len(scope.get('exclusions', []))} items
- Clarifications: {len(scope.get('clarifications', []))} items
- Inclusions: {len(scope.get('inclusions', []))} items

CONTACT:
- Person: {epc.get('contact_person', 'N/A')}
- Email: {epc.get('email', 'N/A')}
- Phone: {epc.get('phone', 'N/A')}
- Proposal Date: {epc.get('proposal_date', 'N/A')}
"""

            # Add detailed SOV if available
            if detailed_sov and any(detailed_sov.values()):
                proposal_context += "\nDETAILED SCHEDULE OF VALUES (SOV):\n"
                proposal_context += self._format_detailed_sov(detailed_sov)

            context_parts.append(proposal_context)

        return "\n".join(context_parts)

    def _format_detailed_sov(self, detailed_sov: Dict) -> str:
        """Format detailed SOV data for chatbot context."""
        formatted = []

        for category, items in detailed_sov.items():
            if not items or not isinstance(items, dict):
                continue

            category_name = category.replace('_', ' ').title()
            formatted.append(f"\n{category_name}:")

            for item_name, item_data in items.items():
                if not isinstance(item_data, dict):
                    continue

                cost = item_data.get('cost')
                unit_cost = item_data.get('unit_cost')

                if cost and cost > 0:
                    item_label = item_name.replace('_', ' ').title()
                    cost_str = f"${cost:,.0f}"
                    unit_str = f" (${unit_cost:.4f}/W)" if unit_cost else ""
                    formatted.append(f"  - {item_label}: {cost_str}{unit_str}")

        return "\n".join(formatted) if formatted else "  No detailed SOV data available"

    def get_detailed_scope(self, proposals: List[Dict], epc_name: str = None) -> str:
        """Get detailed scope information for specific EPC or all EPCs."""
        scope_parts = []

        for proposal in proposals:
            epc = proposal.get('epc_contractor', {}).get('company_name', 'Unknown')

            if epc_name and epc_name.lower() not in epc.lower():
                continue

            scope = proposal.get('scope', {})

            scope_text = f"""
{epc} - DETAILED SCOPE:

ASSUMPTIONS ({len(scope.get('assumptions', []))}):
{chr(10).join(f"- {item}" for item in scope.get('assumptions', [])[:20])}

EXCLUSIONS ({len(scope.get('exclusions', []))}):
{chr(10).join(f"- {item}" for item in scope.get('exclusions', [])[:20])}

CLARIFICATIONS ({len(scope.get('clarifications', []))}):
{chr(10).join(f"- {item}" for item in scope.get('clarifications', [])[:20])}

INCLUSIONS ({len(scope.get('inclusions', []))}):
{chr(10).join(f"- {item}" for item in scope.get('inclusions', [])[:20])}
"""
            scope_parts.append(scope_text)

        return "\n".join(scope_parts) if scope_parts else "No scope information found."

    def chat(self, user_message: str, proposals: List[Dict]) -> str:
        """Process user message and return AI response."""

        # Prepare context
        context = self.prepare_context(proposals)

        # Check if user is asking about scope/assumptions/exclusions
        scope_keywords = ['assumption', 'exclusion', 'clarification', 'inclusion', 'scope', 'included', 'excluded']
        if any(keyword in user_message.lower() for keyword in scope_keywords):
            # Extract EPC name if mentioned
            epc_names = [p.get('epc_contractor', {}).get('company_name', '') for p in proposals]
            mentioned_epc = None
            for epc in epc_names:
                if epc.lower() in user_message.lower():
                    mentioned_epc = epc
                    break

            detailed_scope = self.get_detailed_scope(proposals, mentioned_epc)
            context = f"{context}\n\nDETAILED SCOPE INFORMATION:\n{detailed_scope}"

        # Build conversation with context
        messages = [
            {"role": "system", "content": """You are an expert EPC proposal analyst assistant. You help users understand and compare EPC contractor proposals for renewable energy projects.

Key guidelines:
- Be concise and direct in your answers
- Use specific numbers, costs, and details from the proposals
- When comparing EPCs, provide pros/cons
- If asked about scope items, reference specific assumptions, exclusions, or clarifications
- Format responses clearly with bullet points when appropriate
- If you don't have the information, say so clearly
- When asked for cost breakdowns or comparisons, create markdown tables
- You have access to detailed Schedule of Values (SOV) data including line-item costs for civil, electrical, mechanical, substation, and other categories
- You can filter and aggregate costs by category (e.g., "show me all high voltage costs", "civil work costs", "electrical costs")
- When creating tables, use proper markdown table format with | and alignment
- For cost queries, show both absolute costs ($) and unit costs ($/W) when available

You have access to detailed proposal information including costs, equipment specs, scope details, detailed SOV line items, and more."""},
            {"role": "user", "content": f"Here is the proposal data I have:\n\n{context}"}
        ]

        # Add conversation history (last 5 exchanges)
        messages.extend(self.conversation_history[-10:])

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective for chat
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )

            assistant_message = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message

        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []