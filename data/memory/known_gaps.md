# Known Gaps & Do-Not-Claim Records — AcmeCorp

This document lists capabilities that we DO NOT have and MUST NOT claim in proposals.
These are hard constraints — claiming these capabilities has caused past delivery failures.

## Hard Gaps (Do Not Claim)

### E-Commerce Features
- **No native e-commerce integration**: No Shopify, WooCommerce, Magento, or BigCommerce connectors.
- **No payment processing**: No Stripe, PayPal, Square, or Adyen integration.
- **No product catalog management**: The platform is not designed for retail product management.
- **No real-time inventory synchronization**: No capability for inventory tracking or sync.
- *Reference*: Lost RetailMax proposal (2024-Q1) due to these gaps.

### Federal / Government Compliance
- **No FedRAMP authorization**: Neither moderate nor high baseline. Not on current roadmap.
- **No IL4/IL5 data handling**: Cannot process or store classified/controlled information.
- **No FIPS 140-2 validated cryptography**: Our encryption libraries are not FIPS validated.
- **Missing NIST 800-53 controls**: Several controls required for federal contracts are not implemented.
- *Reference*: Lost GovSecure proposal (2024-Q4). Estimated 12–18 months to achieve FedRAMP.

### Healthcare-Specific Integrations
- **HL7 FHIR integration is NOT standard**: It was custom-built for MediCare project and is not productized. DO NOT claim it as a standard feature. Requires dedicated engineering assessment and 10+ weeks of development.
- **No native EHR/EMR integration**: No direct connectors to Epic, Cerner, or Allscripts.
- **No DICOM support**: Cannot process or store medical imaging data.

### Mobile & Native App
- **No native mobile application**: Only responsive web interface available. No iOS or Android native app.
- **No offline mode**: The platform requires an active internet connection. No offline data sync capability.
- **No push notification infrastructure**: Push notifications are not supported (only email and in-app).

### Advanced AI/ML Features
- **No built-in AI/ML capabilities**: No native machine learning, predictive analytics, or NLP features.
- **No recommendation engine**: Cannot provide automated recommendations or suggestions.
- **No computer vision or image processing**: Not applicable to our platform.

### Real-Time Features
- **No real-time collaboration**: No concurrent editing (like Google Docs). Changes require page refresh.
- **No WebSocket-based live updates**: Real-time dashboards use polling (30-second minimum interval), not true real-time.
- **No video conferencing or live streaming**: Not a communication platform.

## Soft Gaps (Proceed With Caution)

### Performance at Scale
- **>5,000 concurrent users**: Not tested above 5,000 concurrent users. Enterprise tier supports up to 5,000. Larger deployments require custom infrastructure assessment. DO NOT commit to >5,000 without engineering sign-off.
- **>10M row datasets for analytics**: Analytics API SLA is for datasets up to 10M rows. Performance degrades significantly beyond this threshold. Requires data partitioning strategy.

### International / Localization
- **Limited i18n support**: UI is English-only. Content can be in any language but UI labels, error messages, and system notifications are English only. Full i18n is on the roadmap but not scheduled.
- **Limited timezone handling**: System timestamps are UTC. Timezone display conversion is available but some reports default to UTC.

### Deprecated Features (Do Not Claim)
- **SMS-based MFA**: Deprecated in v4.0, scheduled for removal in v5.0. Do not include in new proposals.
- **SOAP API**: Legacy SOAP API is in maintenance mode. Not recommended for new integrations. Will be removed in v6.0.
- **Internet Explorer support**: Dropped as of v4.0. Supported browsers: Chrome, Firefox, Safari, Edge (Chromium).
