# Past Proposals & Outcomes — AcmeCorp

## Proposal: GlobalBank Enterprise Deployment (2024-Q3)
- **Outcome**: WON
- **Contract Value**: $1.2M annual
- **Key Requirements Met**:
  - SSO via SAML 2.0 with Azure AD — delivered as claimed.
  - RBAC with 3 custom role hierarchies — delivered with minor configuration.
  - SOC 2 Type II compliance — already certified.
  - Data encryption at rest (AES-256) — standard feature.
  - Audit logging with 7-year retention — Enterprise tier feature.
- **Delivery Notes**:
  - Custom SAML attribute mapping required 2 weeks of professional services (not in original estimate).
  - Client requested LDAP integration which was not in the proposal — handled via post-contract change order.
  - Actual deployment took 8 weeks vs estimated 6 weeks due to client's internal security review delays.
- **Lessons Learned**:
  - Always include professional services buffer for SAML/SSO customizations.
  - Client security review timelines should be factored into deployment estimates.

## Proposal: MediCare Health Platform (2024-Q2)
- **Outcome**: WON (with conditions)
- **Contract Value**: $800K annual
- **Key Requirements Met**:
  - HIPAA compliance with BAA — available on Enterprise tier.
  - Data residency in US region — supported.
  - Field-level encryption for PHI — available since v3.8.
  - Webhook notifications for appointment events — standard feature.
- **Issues During Delivery**:
  - Client required HL7 FHIR integration — this was NOT available and required 3 months of custom development by a dedicated team.
  - Performance issues when handling 10,000+ concurrent users; required infrastructure upgrade beyond original scope.
  - Custom audit report format required significant professional services.
- **Actual Effort vs Estimated**:
  - Estimated: 12 weeks implementation. Actual: 22 weeks.
  - HL7 FHIR integration alone consumed 10 weeks of engineering effort.
  - Total cost overrun: approximately 40% above original budget.
- **Lessons Learned**:
  - Healthcare integrations (HL7, FHIR) are NOT standard — do not promise without engineering assessment.
  - Performance testing for high-concurrency healthcare scenarios should be mandatory before commitment.

## Proposal: RetailMax E-Commerce Integration (2024-Q1)
- **Outcome**: LOST
- **Reason**: Competitor offered native e-commerce platform integration we could not match.
- **Key Gaps Identified**:
  - No native Shopify or WooCommerce connector.
  - No support for real-time inventory sync.
  - Payment processing integration (Stripe, PayPal) not available.
  - Product catalog management features do not exist in our platform.
- **Competitor**: SolutionWorks Inc. — won with a specialized e-commerce vertical solution.
- **Lessons Learned**:
  - E-commerce vertical is outside our platform's sweet spot. Do not pursue unless client's e-commerce needs are secondary.
  - Sales team should qualify e-commerce requirements early and deprioritize if they are core needs.

## Proposal: GovSecure Federal Contract (2024-Q4)
- **Outcome**: LOST
- **Reason**: FedRAMP certification required; we are not FedRAMP authorized.
- **Key Gaps**:
  - No FedRAMP authorization (moderate or high baseline).
  - No IL4/IL5 data handling capability.
  - No FIPS 140-2 validated cryptographic modules.
  - Missing certain NIST 800-53 controls.
- **Effort Estimate If Pursued**: FedRAMP authorization estimated at 12–18 months and $500K–$1M in compliance and infrastructure investment.
- **Lessons Learned**:
  - FedRAMP is a hard requirement for US federal contracts. Do not bid without it.
  - FedRAMP roadmap item has been proposed to product leadership but not yet approved.

## Proposal: EduTech University SaaS (2025-Q1)
- **Outcome**: WON
- **Contract Value**: $200K annual
- **Key Requirements Met**:
  - Multi-tenant SaaS with department-level isolation — standard feature.
  - OIDC SSO with Google Workspace — supported natively.
  - CSV bulk import of student records (50,000+) — bulk API supports 100K per batch.
  - Dashboard builder for enrollment analytics — custom dashboard feature.
- **Delivery Notes**:
  - Smooth delivery. Completed in 4 weeks vs estimated 5 weeks.
  - Client was very satisfied with dashboard customization capabilities.
  - Minor issue: CSV import validation error messages were not user-friendly; fixed in v4.3.
- **Lessons Learned**:
  - Education sector is a strong fit for our platform. Repeat this pattern.
  - Bulk data import UX should be improved (product backlog item #PLT-2847).
